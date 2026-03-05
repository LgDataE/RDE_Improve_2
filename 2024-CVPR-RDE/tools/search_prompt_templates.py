import argparse
import json
import os
import os.path as op
import random
import time
import urllib.request

import torch
import torch.nn.functional as F

from easydict import EasyDict as edict

from datasets.bases import ImageDataset, TextDataset
from datasets.build import build_transforms
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.metrics import get_metrics


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            text = text.lstrip()
            if text.startswith("json"):
                text = text[4:]
    return text.strip()


def _extract_json_array(text: str):
    text = _strip_code_fence(text)
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _openai_chat(api_key: str, base_url: str, model: str, messages, temperature: float, max_tokens: int, timeout: int):
    url = base_url.rstrip('/') + '/chat/completions'
    payload = {
        'model': model,
        'messages': messages,
        'temperature': float(temperature),
        'max_tokens': int(max_tokens),
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8')
    obj = json.loads(body)
    return obj['choices'][0]['message']['content']


def _generate_templates(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    n: int,
    extra_instructions: str,
):
    sys_prompt = (
        "You generate prompt templates for text-to-image person re-identification using CLIP-style text encoder. "
        "Templates will be used to wrap a caption. "
        "Return only JSON."
    )
    user_prompt = (
        "Generate {n} diverse templates. Requirements:\n"
        "- Each template MUST contain the substring {caption} exactly once\n"
        "- Keep templates short (<= 15 words before {caption})\n"
        "- Avoid adding new facts; templates should be generic\n"
        "- Mix neutral and photo-style phrasing\n"
        "- Output ONLY a JSON array of strings\n"
        "{extra}\n"
    ).format(n=int(n), extra=str(extra_instructions or "").strip())

    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    content = _openai_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    arr = _extract_json_array(content)
    if not isinstance(arr, list):
        raise RuntimeError('LLM did not return a JSON array')

    cleaned = []
    seen = set()
    for t in arr:
        s = str(t).strip()
        if not s:
            continue
        if '{caption}' not in s:
            continue
        s = ' '.join(s.split())
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


def _build_dataset(args):
    if args.dataset_name == 'RSTPReid':
        return RSTPReid(root=args.root_dir, anno_path=getattr(args, 'rstpreid_anno_path', ''))
    if args.dataset_name == 'CUHK-PEDES':
        return CUHKPEDES(root=args.root_dir)
    if args.dataset_name == 'ICFG-PEDES':
        return ICFGPEDES(root=args.root_dir)
    raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")


def _compute_image_feats(model, img_loader, use_tse: bool):
    model = model.eval()
    device = next(model.parameters()).device
    gids, gfeats = [], []
    for pid, img in img_loader:
        img = img.to(device)
        with torch.no_grad():
            feat = model.encode_image_tse(img) if use_tse else model.encode_image(img)
        gids.append(pid.view(-1).cpu())
        gfeats.append(feat.cpu())
    gids = torch.cat(gids, 0)
    gfeats = torch.cat(gfeats, 0)
    gfeats = F.normalize(gfeats, p=2, dim=1)
    return gfeats, gids


def _compute_text_feats(model, txt_loader, use_tse: bool):
    model = model.eval()
    device = next(model.parameters()).device
    qids, qfeats = [], []
    for pid, caption in txt_loader:
        caption = caption.to(device)
        with torch.no_grad():
            if caption.dim() == 3:
                b, t, l = caption.shape
                caption_flat = caption.reshape(b * t, l)
                feat = model.encode_text_tse(caption_flat) if use_tse else model.encode_text(caption_flat)
                feat = feat.reshape(b, t, -1).mean(dim=1).cpu()
            else:
                feat = (model.encode_text_tse(caption) if use_tse else model.encode_text(caption)).cpu()
        qids.append(pid.view(-1).cpu())
        qfeats.append(feat.cpu())
    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)
    qfeats = F.normalize(qfeats, p=2, dim=1)
    return qfeats, qids


def _proxy_margin_score(sim, qids, gids):
    sim = sim.float()
    qids = qids.view(-1)
    gids = gids.view(-1)
    same = qids[:, None].eq(gids[None, :])

    sim_pos = sim.masked_fill(~same, float('-inf')).max(dim=1).values
    sim_neg = sim.masked_fill(same, float('-inf')).max(dim=1).values

    margin = sim_pos - sim_neg
    ok = torch.isfinite(margin)
    if ok.any():
        return float(margin[ok].mean().item())
    return float('-inf')


def _proxy_r1(sim, qids, gids):
    sim = sim.float()
    pred = gids[sim.argmax(dim=1)]
    return float(pred.eq(qids).float().mean().item())


def _eval_bge_tse(q_bge, g_bge, q_tse, g_tse, qids, gids):
    sims_bge = q_bge @ g_bge.t()
    sims_tse = q_tse @ g_tse.t()
    sims = (sims_bge + sims_tse) / 2
    rs = get_metrics(sims, qids, gids, 'BGE+TSE-t2i', False)
    return {
        'R1': float(rs[1]),
        'R5': float(rs[2]),
        'R10': float(rs[3]),
        'mAP': float(rs[4]),
        'mINP': float(rs[5]),
        'rSum': float(rs[6]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--rstpreid_anno_path', type=str, default='')
    parser.add_argument('--val_dataset', type=str, default='val', choices=['val', 'test'])

    parser.add_argument('--out_dir', type=str, required=True)

    parser.add_argument('--provider', type=str, default='openai', choices=['openai'])
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--timeout', type=int, default=60)

    parser.add_argument('--generate_n', type=int, default=0)
    parser.add_argument('--template_extra', type=str, default='')
    parser.add_argument('--in_templates_json', type=str, default='')

    parser.add_argument('--seed_templates', type=str, nargs='*', default=[])

    parser.add_argument('--proxy_metric', type=str, default='margin', choices=['margin', 'r1'])
    parser.add_argument('--proxy_topk', type=int, default=20)
    parser.add_argument('--eval_topk', type=int, default=5)

    parser.add_argument('--proxy_max_queries', type=int, default=2000)
    parser.add_argument('--proxy_seed', type=int, default=0)

    parser.add_argument('--sleep', type=float, default=0.0)

    args_cli = parser.parse_args()

    os.makedirs(args_cli.out_dir, exist_ok=True)

    args = load_train_configs(args_cli.config_file)
    args.training = False
    args.val_dataset = args_cli.val_dataset

    if args_cli.rstpreid_anno_path:
        args.rstpreid_anno_path = args_cli.rstpreid_anno_path

    dataset = _build_dataset(args)

    ds = dataset.val if args_cli.val_dataset == 'val' else dataset.test

    test_transforms = build_transforms(img_size=args.img_size, is_train=False)
    img_set = ImageDataset(ds['image_pids'], ds['img_paths'], test_transforms, args=args)

    q_pids = list(ds['caption_pids'])
    q_caps = list(ds['captions'])

    if args_cli.proxy_max_queries and int(args_cli.proxy_max_queries) > 0 and len(q_caps) > int(args_cli.proxy_max_queries):
        rng = random.Random(int(args_cli.proxy_seed))
        idxs = list(range(len(q_caps)))
        rng.shuffle(idxs)
        idxs = idxs[: int(args_cli.proxy_max_queries)]
        q_pids = [q_pids[i] for i in idxs]
        q_caps = [q_caps[i] for i in idxs]

    num_classes = len(dataset.train_id_container)
    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=args_cli.ckpt)
    model = model.cuda()

    img_loader = torch.utils.data.DataLoader(
        img_set,
        batch_size=int(getattr(args, 'test_batch_size', 512)),
        shuffle=False,
        num_workers=int(getattr(args, 'num_workers', 8)),
        pin_memory=True,
        drop_last=False,
    )

    g_bge, gids = _compute_image_feats(model, img_loader, use_tse=False)
    g_tse, _ = _compute_image_feats(model, img_loader, use_tse=True)

    templates = []
    for t in args_cli.seed_templates:
        s = str(t).strip()
        if s and s not in templates:
            templates.append(s)

    if args_cli.in_templates_json:
        with open(args_cli.in_templates_json, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise RuntimeError('--in_templates_json must be a JSON array')
        for t in loaded:
            s = str(t).strip()
            if s and s not in templates:
                templates.append(s)

    if int(args_cli.generate_n) > 0:
        api_key = args_cli.api_key or os.environ.get('OPENAI_API_KEY', '')
        if not api_key:
            raise RuntimeError('Missing API key. Set OPENAI_API_KEY or pass --api_key')
        gen = _generate_templates(
            api_key=api_key,
            base_url=args_cli.base_url,
            model=args_cli.model,
            temperature=args_cli.temperature,
            max_tokens=args_cli.max_tokens,
            timeout=args_cli.timeout,
            n=int(args_cli.generate_n),
            extra_instructions=args_cli.template_extra,
        )
        for t in gen:
            if t not in templates:
                templates.append(t)
        if float(args_cli.sleep) > 0:
            time.sleep(float(args_cli.sleep))

    if not templates:
        raise RuntimeError('No templates provided/generated')

    results = []

    for template in templates:
        tmp_args = edict(dict(args))
        tmp_args.prompt_template = str(template)
        tmp_args.prompt_templates = []
        tmp_args.prompt_ensemble = False

        txt_set = TextDataset(q_pids, q_caps, text_length=args.text_length, args=tmp_args)
        txt_loader = torch.utils.data.DataLoader(
            txt_set,
            batch_size=int(getattr(args, 'test_batch_size', 512)),
            shuffle=False,
            num_workers=int(getattr(args, 'num_workers', 8)),
            pin_memory=True,
            drop_last=False,
        )

        q_bge, qids = _compute_text_feats(model, txt_loader, use_tse=False)
        q_tse, _ = _compute_text_feats(model, txt_loader, use_tse=True)

        sims = (q_bge @ g_bge.t() + q_tse @ g_tse.t()) / 2
        if args_cli.proxy_metric == 'r1':
            proxy = _proxy_r1(sims, qids, gids)
        else:
            proxy = _proxy_margin_score(sims, qids, gids)

        results.append({
            'template': template,
            'proxy': float(proxy),
        })

    results.sort(key=lambda x: x['proxy'], reverse=True)
    proxy_topk = max(1, int(args_cli.proxy_topk))
    eval_topk = max(1, int(args_cli.eval_topk))

    top_for_eval = results[:proxy_topk]

    for rec in top_for_eval[:eval_topk]:
        template = rec['template']
        tmp_args = edict(dict(args))
        tmp_args.prompt_template = str(template)
        tmp_args.prompt_templates = []
        tmp_args.prompt_ensemble = False

        txt_set = TextDataset(q_pids, q_caps, text_length=args.text_length, args=tmp_args)
        txt_loader = torch.utils.data.DataLoader(
            txt_set,
            batch_size=int(getattr(args, 'test_batch_size', 512)),
            shuffle=False,
            num_workers=int(getattr(args, 'num_workers', 8)),
            pin_memory=True,
            drop_last=False,
        )

        q_bge, qids = _compute_text_feats(model, txt_loader, use_tse=False)
        q_tse, _ = _compute_text_feats(model, txt_loader, use_tse=True)

        rec['eval'] = _eval_bge_tse(q_bge, g_bge, q_tse, g_tse, qids, gids)

    out_results = {
        'config_file': args_cli.config_file,
        'ckpt': args_cli.ckpt,
        'val_dataset': args_cli.val_dataset,
        'proxy_metric': args_cli.proxy_metric,
        'templates': results,
    }

    with open(op.join(args_cli.out_dir, 'template_search_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out_results, f, indent=2, ensure_ascii=False)

    final_candidates = []
    for rec in results:
        if 'eval' in rec and isinstance(rec['eval'], dict) and 'rSum' in rec['eval']:
            final_candidates.append(rec)

    if final_candidates:
        final_candidates.sort(key=lambda x: x['eval']['rSum'], reverse=True)
        best = [r['template'] for r in final_candidates[:eval_topk]]
    else:
        best = [r['template'] for r in results[:eval_topk]]

    with open(op.join(args_cli.out_dir, 'prompt_templates.json'), 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    with open(op.join(args_cli.out_dir, 'prompt_templates.txt'), 'w', encoding='utf-8') as f:
        for t in best:
            f.write(str(t).strip() + '\n')

    print('Saved:')
    print(op.join(args_cli.out_dir, 'template_search_results.json'))
    print(op.join(args_cli.out_dir, 'prompt_templates.json'))
    print(op.join(args_cli.out_dir, 'prompt_templates.txt'))


if __name__ == '__main__':
    main()
