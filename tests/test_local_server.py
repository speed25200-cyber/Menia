import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from menia.local_server import LocalServerGenerator


class LocalServerTests(unittest.TestCase):
    def setUp(self):
        self.received = []
        self.content = 'Information reçue, non observée.'
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def do_GET(self):
                self.reply({'data': [{'id': 'test-model'}]})

            def do_POST(self):
                owner.received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.reply({'choices': [{'message': {'content': owner.content}, 'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 20, 'completion_tokens': 8}})

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_transport_preserves_evidence_and_roles(self):
        generator = LocalServerGenerator(self.url)
        messages = [{'role': 'system', 'content': 'Témoignage #7, non vérifié'},
                    {'role': 'user', 'content': 'Que sais-tu ?'}]
        self.assertEqual(generator(messages), self.content)
        self.assertEqual(self.received[0]['messages'], messages)
        self.assertFalse(self.received[0]['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(generator.last_metrics['output_tokens'], 8)

    def test_empty_response_is_rejected(self):
        generator = LocalServerGenerator(self.url)
        self.content = None
        with self.assertRaises(ValueError):
            generator([{'role': 'user', 'content': 'Bonjour'}])

    def test_nonlocal_or_credentialed_urls_are_rejected(self):
        for url in ('https://example.com', 'http://example.com', 'http://user:pass@localhost',
                    'http://localhost/v1', 'http://localhost?secret=1'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalServerGenerator(url)
