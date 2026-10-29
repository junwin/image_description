import json
from unittest.mock import Mock

import pytest

from image_description.cli import instagram_cli, instagram_comments


def test_paginated_posts_comments_and_replies_are_read_only(monkeypatch):
    calls = []

    def api(method, path, token, **kwargs):
        assert method == 'GET'
        assert token == 'secret'
        params = kwargs['params']
        calls.append((path, params))
        after = params.get('after')
        pages = {
            ('user/media', None): {'data': [{'id': 'p1', 'caption': 'First'}],
                'paging': {'cursors': {'after': 'posts2'}, 'next': 'https://evil.example/?access_token=secret'}},
            ('user/media', 'posts2'): {'data': [{'id': 'p2', 'media_type': 'VIDEO'}]},
            ('p1/comments', None): {'data': [{'id': 'c1', 'text': 'Nice', 'username': 'alice'}],
                'paging': {'cursors': {'after': 'comments2'}, 'next': 'unused'}},
            ('p1/comments', 'comments2'): {'data': [{'id': 'c2', 'text': 'Lovely'}]},
            ('c1/replies', None): {'data': [{'id': 'r1', 'text': 'Thanks'}],
                'paging': {'cursors': {'after': 'replies2'}, 'next': 'unused'}},
            ('c1/replies', 'replies2'): {'data': [{'id': 'r2', 'text': 'Indeed'}]},
            ('c2/replies', None): {'data': []},
            ('p2/comments', None): {'data': []},
        }
        return pages[(path, after)]

    monkeypatch.setattr(instagram_cli, '_api', api)
    report = instagram_comments.fetch_comments('user', 'secret', posts=2)
    assert [p['id'] for p in report['posts']] == ['p1', 'p2']
    comments = report['posts'][0]['comments']
    assert [c['id'] for c in comments] == ['c1', 'c2']
    assert [r['id'] for r in comments[0]['replies']] == ['r1', 'r2']
    assert report['posts'][1]['comments'] == []
    assert len(calls) == 8
    assert all(not path.startswith('http') for path, _ in calls)


def test_default_reads_only_ten_posts_and_can_skip_replies(monkeypatch):
    calls = []

    def api(method, path, token, **kwargs):
        calls.append((method, path, kwargs['params']))
        if path == 'u/media':
            assert kwargs['params']['limit'] == 10
            return {'data': [{'id': f'p{i}'} for i in range(12)],
                    'paging': {'next': 'unused', 'cursors': {'after': 'more'}}}
        return {'data': [{'id': path + '-comment', 'text': 'A comment'}]}

    monkeypatch.setattr(instagram_cli, '_api', api)
    report = instagram_comments.fetch_comments('u', 'secret', include_replies=False)
    assert len(report['posts']) == 10
    assert len(calls) == 11
    assert all(method == 'GET' for method, _, _ in calls)
    assert all('replies' not in p['comments'][0] for p in report['posts'])


@pytest.mark.parametrize('posts', [0, -1])
def test_invalid_count_before_credentials_or_network(monkeypatch, posts):
    load = Mock(side_effect=AssertionError('credentials read'))
    monkeypatch.setattr(instagram_cli, 'load_credentials', load)
    with pytest.raises(ValueError, match='positive integer'):
        instagram_comments.cmd_comments(posts=posts)
    load.assert_not_called()


def test_text_output_includes_post_details_comments_and_replies(monkeypatch, capsys):
    monkeypatch.setattr(instagram_cli, 'load_credentials', lambda path: ('u', 'secret'))
    monkeypatch.setattr(instagram_comments, 'fetch_comments', lambda *a, **kw: {
        'instagram_user_id': 'u', 'posts': [
            {'id': 'p1', 'timestamp': '2026-10-28', 'caption': 'A photograph',
             'permalink': 'https://www.instagram.com/p/example/', 'comments': [
                {'id': 'c1', 'username': 'alice', 'text': 'Beautiful\nColours',
                 'replies': [{'id': 'r1', 'text': 'Thank you'}]}]},
            {'id': 'p2', 'comments': []},
        ]})
    instagram_comments.cmd_comments()
    output = capsys.readouterr().out
    for text in ('A photograph', '2026-10-28', 'https://www.instagram.com/p/example/',
                 'alice', 'Beautiful', 'Colours', 'Thank you', 'unknown author', 'No comments returned'):
        assert text in output
    assert 'secret' not in output


def test_main_json_and_credentials_flags(monkeypatch, capsys):
    load = Mock(return_value=('u', 'secret'))
    fetch = Mock(return_value={'instagram_user_id': 'u', 'posts': []})
    monkeypatch.setattr(instagram_cli, 'load_credentials', load)
    monkeypatch.setattr(instagram_comments, 'fetch_comments', fetch)
    instagram_cli.main(['comments', '--posts', '3', '--json', '--no-replies', '--credentials', '/tmp/ig.json'])
    load.assert_called_once_with('/tmp/ig.json')
    fetch.assert_called_once_with('u', 'secret', posts=3, include_replies=False)
    assert json.loads(capsys.readouterr().out) == {'instagram_user_id': 'u', 'posts': []}


def test_no_posts_text_output(monkeypatch, capsys):
    monkeypatch.setattr(instagram_cli, 'load_credentials', lambda path: ('u', 'secret'))
    monkeypatch.setattr(instagram_cli, '_api', lambda *a, **kw: {'data': []})
    instagram_comments.cmd_comments()
    assert 'No Instagram posts returned' in capsys.readouterr().out


def test_permission_error_is_nonzero_clear_and_not_a_partial_report(monkeypatch, capsys):
    monkeypatch.setattr(instagram_cli, 'load_credentials', lambda path: ('u', 'secret'))

    def api(method, path, token, **kwargs):
        if path == 'u/media':
            return {'data': [{'id': 'p1'}]}
        raise RuntimeError('Instagram API HTTP 403: Permissions error (code 200)')

    monkeypatch.setattr(instagram_cli, '_api', api)
    with pytest.raises(SystemExit) as exc:
        instagram_cli.main(['comments', '--json'])
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out == ''
    assert 'instagram_business_manage_comments' in output.err
    assert 'secret' not in output.err


def test_expired_token_error_is_not_reported_as_missing_comments(monkeypatch, capsys):
    monkeypatch.setattr(instagram_cli, 'load_credentials', lambda path: ('u', 'secret'))
    monkeypatch.setattr(instagram_cli, '_api', Mock(side_effect=RuntimeError(
        'Instagram API HTTP 400: Token expired (code 190)')))
    with pytest.raises(SystemExit):
        instagram_cli.main(['comments'])
    output = capsys.readouterr()
    assert output.out == ''
    assert 'Token expired' in output.err
    assert 'instagram_business_manage_comments' not in output.err


@pytest.mark.parametrize('paging', [
    {'next': 'unused'},
    {'next': 'unused', 'cursors': {'after': 'same'}},
])
def test_broken_pagination_fails_instead_of_returning_incomplete_comments(monkeypatch, paging):
    monkeypatch.setattr(instagram_cli, '_api', lambda *a, **kw: {'data': [], 'paging': paging})
    with pytest.raises(RuntimeError, match='pagination'):
        list(instagram_comments._iter_edge('p/comments', 'secret', 'id,text'))


def test_duplicate_items_are_not_printed_twice(monkeypatch):
    responses = iter([
        {'data': [{'id': 'c1'}], 'paging': {'next': 'unused', 'cursors': {'after': 'more'}}},
        {'data': [{'id': 'c1'}, {'id': 'c2'}]},
    ])
    monkeypatch.setattr(instagram_cli, '_api', lambda *a, **kw: next(responses))
    assert [c['id'] for c in instagram_comments._iter_edge('p/comments', 'secret', 'id')] == ['c1', 'c2']


def test_request_failure_surfaces_without_empty_report(monkeypatch, capsys):
    monkeypatch.setattr(instagram_cli, 'load_credentials', lambda path: ('u', 'secret'))
    monkeypatch.setattr(instagram_cli.requests, 'request', Mock(side_effect=instagram_cli.requests.Timeout('timeout')))
    with pytest.raises(SystemExit):
        instagram_cli.main(['comments'])
    output = capsys.readouterr()
    assert output.out == ''
    assert 'timeout' in output.err
