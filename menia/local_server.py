"""Standard-library client for a local llama.cpp language server."""
import json
import time
from urllib.parse import urlparse
from urllib.request import Request, ProxyHandler, build_opener


class LocalServerGenerator:
    def __init__(self, url, *, timeout=60):
        parsed = urlparse(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1')
                or parsed.path not in ('', '/') or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise ValueError('Use the root HTTP URL of a local loopback server')
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))
        with self.opener.open(self.url+'/v1/models', timeout=timeout) as response:
            models = json.load(response)['data']
        if len(models) != 1:
            raise ValueError('A single explicitly loaded local model is required')
        self.model_id = models[0]['id']
        self.last_metrics = None

    def __call__(self, messages):
        body = {'model': self.model_id, 'messages': messages, 'stream': False,
                'temperature': 0, 'seed': 17, 'max_tokens': 192,
                'chat_template_kwargs': {'enable_thinking': False}}
        request = Request(self.url+'/v1/chat/completions',
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        started = time.perf_counter()
        with self.opener.open(request, timeout=self.timeout) as response:
            value = json.load(response)
        choice = value['choices'][0]
        content = choice['message'].get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('The language server returned no textual answer')
        usage = value.get('usage', {})
        self.last_metrics = {'backend': 'llama.cpp', 'model': self.model_id,
            'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'),
            'seconds': time.perf_counter()-started,
            'reached_token_limit': choice.get('finish_reason') == 'length'}
        return content.strip()
