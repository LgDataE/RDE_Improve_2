import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request


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


def _load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: str, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def _load_cache_jsonl(cache_path: str):
    cache = {}
    if not cache_path or not os.path.exists(cache_path):
        return cache
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (str(obj.get('split', '')), str(obj.get('id', '')), str(obj.get('img_path', '')))
            cache[key] = obj.get('captions', [])
    return cache


def _append_cache_jsonl(cache_path: str, split: str, pid, img_path: str, captions):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    rec = {
        'split': split,
        'id': pid,
        'img_path': img_path,
        'captions': captions,
    }
    with open(cache_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _generate_captions_for_item(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    instruction_template: str,
    in_captions,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
):
    last_err = None
    captions_json = json.dumps(in_captions, ensure_ascii=False)
    user_prompt = instruction_template.replace('{captions}', captions_json)
    for attempt in range(max_retries + 1):
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
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
            out = _extract_json_array(content)
            if not isinstance(out, list):
                raise ValueError('LLM output is not a JSON array')
            out = [str(x).strip() for x in out]
            return out
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(max(0.0, float(retry_sleep)) * (2 ** attempt))
                continue
            raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_json', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--cache_jsonl', type=str, default='')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--provider', type=str, default='openai', choices=['openai'])
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1')
    parser.add_argument('--api_key', type=str, default='')

    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--timeout', type=int, default=60)

    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--retry_sleep', type=float, default=2.0)
    parser.add_argument('--sleep', type=float, default=0.0)

    parser.add_argument('--splits', type=str, nargs='*', default=['train', 'val', 'test'])
    parser.add_argument('--max_items', type=int, default=0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--system_prompt', type=str, default='You rewrite person descriptions for text-based person re-identification. Keep only information that is explicitly present. Do not add new facts. Output only valid JSON.')
    parser.add_argument(
        '--instruction',
        type=str,
        default='Rewrite each caption below into one concise sentence for person re-identification. Preserve meaning, normalize grammar, keep discriminative appearance attributes (clothing colors/patterns, accessories, carried objects, hair, gender if stated). Do not hallucinate. Return ONLY a JSON array of strings with the same length and order as the input. Input captions: {captions}',
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY', '')
    if args.provider == 'openai' and not api_key:
        raise RuntimeError('Missing API key. Set OPENAI_API_KEY or pass --api_key')

    annos = _load_json(args.in_json)
    if not isinstance(annos, list):
        raise RuntimeError('Input JSON must be a list of annotations')

    cache = {}
    if args.resume and args.cache_jsonl:
        cache = _load_cache_jsonl(args.cache_jsonl)

    candidate = []
    for idx, anno in enumerate(annos):
        split = str(anno.get('split', ''))
        if split in set(args.splits):
            candidate.append(idx)

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(candidate)

    if int(args.max_items) > 0:
        candidate = candidate[: int(args.max_items)]

    for i, idx in enumerate(candidate):
        anno = annos[idx]
        split = str(anno.get('split', ''))
        pid = anno.get('id', None)
        img_path = str(anno.get('img_path', ''))
        key = (split, str(pid), img_path)

        if key in cache:
            continue

        in_captions = anno.get('captions', [])
        if not isinstance(in_captions, list) or not in_captions:
            out_captions = in_captions
        else:
            out_captions = _generate_captions_for_item(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                system_prompt=args.system_prompt,
                instruction_template=args.instruction,
                in_captions=in_captions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            if len(out_captions) != len(in_captions):
                if len(out_captions) > len(in_captions):
                    out_captions = out_captions[: len(in_captions)]
                else:
                    out_captions = list(out_captions) + list(in_captions[len(out_captions) :])

        _append_cache_jsonl(args.cache_jsonl, split, pid, img_path, out_captions)
        cache[key] = out_captions

        if float(args.sleep) > 0:
            time.sleep(float(args.sleep))

    out_annos = []
    for anno in annos:
        split = str(anno.get('split', ''))
        pid = anno.get('id', None)
        img_path = str(anno.get('img_path', ''))
        key = (split, str(pid), img_path)
        new_anno = dict(anno)
        if key in cache:
            new_anno['captions'] = cache[key]
        out_annos.append(new_anno)

    _write_json(args.out_json, out_annos)


if __name__ == '__main__':
    main()
