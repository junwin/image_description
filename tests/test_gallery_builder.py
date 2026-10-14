import json
from pathlib import Path

import pytest

from image_description.cli.gallery_builder_cli import main
from image_description.gallery import build_gallery


def add_image(folder, name, **metadata):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b'image')
    data = {'image_filename': name, 'image_description': 'A visible object.',
            'can_publish': False, 'publish_history': [{'platform': 'mastodon', 'post_id': '123'}],
            'hashtags': '#boutique', 'custom': {'keep': True}}
    data.update(metadata)
    path = (folder / name).with_suffix('.json')
    path.write_text(json.dumps(data), encoding='utf-8')
    return path, data


def test_folder_export_is_compact_and_preserves_source_sidecars(tmp_path):
    folder = tmp_path / 'assets' / 'Très Chic Accessories'
    sidecar, original = add_image(folder, 'z.jpg', image_relative_path='assets/Très Chic Accessories/z.jpg')
    add_image(folder, 'A.png', can_publish=True)
    original_bytes = sidecar.read_bytes()
    (folder / 'ignore.social.json').write_text('{}')
    main(['assets/Très Chic Accessories', '--image-root', str(tmp_path)])
    records = json.loads((folder / 'gallery.json').read_text())
    assert records == [{'file': 'A.png', 'alt': 'A visible object.'},
                       {'file': 'z.jpg', 'alt': 'A visible object.'}]
    assert sidecar.read_bytes() == original_bytes
    # Existing gallery.json is not itself selected on subsequent runs.
    main(['assets/Très Chic Accessories', '--image-root', str(tmp_path)])
    assert len(json.loads((folder / 'gallery.json').read_text())) == 2


def test_selection_replaces_only_gallery_in_requested_order(tmp_path):
    for name in ['a.jpg', 'b.jpg', 'c.jpg']:
        add_image(tmp_path, name)
    main([str(tmp_path)])
    main([str(tmp_path), '--images', 'c.jpg', 'a.jpg'])
    records = json.loads((tmp_path / 'gallery.json').read_text())
    assert [r['file'] for r in records] == ['c.jpg', 'a.jpg']
    assert (tmp_path / 'b.jpg').exists()
    assert (tmp_path / 'b.json').exists()


@pytest.mark.parametrize('problem', ['missing_sidecar', 'blank_alt', 'bad_json', 'wrong_identity', 'wrong_root', 'empty', 'duplicate', 'escape', 'stem_collision'])
def test_invalid_inputs_preserve_previous_gallery(tmp_path, problem):
    sidecar, _ = add_image(tmp_path, 'a.jpg')
    previous = '[{"file":"previous.jpg","alt":"Previous"}]\n'
    output = tmp_path / 'gallery.json'
    output.write_text(previous)
    args = [str(tmp_path)]
    if problem == 'missing_sidecar':
        sidecar.unlink()
    elif problem == 'blank_alt':
        sidecar.write_text('{"image_description":"   "}')
    elif problem == 'bad_json':
        sidecar.write_text('{')
    elif problem == 'wrong_identity':
        add_image(tmp_path, 'a.jpg', image_filename='other.png')
    elif problem == 'wrong_root':
        add_image(tmp_path, 'a.jpg', image_relative_path='../a.jpg')
    elif problem == 'empty':
        (tmp_path / 'a.jpg').unlink()
    elif problem == 'duplicate':
        args += ['--images', 'a.jpg', 'a.jpg']
    elif problem == 'escape':
        args += ['--images', '../outside.jpg']
    elif problem == 'stem_collision':
        (tmp_path / 'a.png').write_bytes(b'image')
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert output.read_text() == previous


def test_legacy_alt_and_path_in_image_filename(tmp_path):
    add_image(tmp_path, 'a.jpg', image_filename='older/path/a.jpg', image_description='',
              visually_challenged_description='Legacy alt')
    record = build_gallery(str(tmp_path))[0]
    assert record['alt'] == 'Legacy alt'
    assert record == {'file': 'a.jpg', 'alt': 'Legacy alt'}


def test_folder_and_image_symlink_escape(tmp_path):
    root = tmp_path / 'web'
    root.mkdir()
    outside = tmp_path / 'outside'
    add_image(outside, 'a.jpg')
    with pytest.raises(ValueError, match='escapes'):
        build_gallery('../outside', image_root=str(root))
    (root / 'a.jpg').symlink_to(outside / 'a.jpg')
    with pytest.raises(ValueError, match='escapes'):
        build_gallery(str(root))


def test_gallery_named_image_is_not_overwritten(tmp_path):
    sidecar, _ = add_image(tmp_path, 'gallery.jpg')
    previous = sidecar.read_bytes()
    with pytest.raises(SystemExit):
        main([str(tmp_path)])
    assert sidecar.read_bytes() == previous
    # The reserved image's sidecar is protected even when it is not selected.
    add_image(tmp_path, 'other.jpg')
    with pytest.raises(SystemExit):
        main([str(tmp_path), '--images', 'other.jpg'])
    assert sidecar.read_bytes() == previous


@pytest.mark.parametrize('use_root', [False, True])
def test_social_directory_processing_ignores_gallery(tmp_path, use_root):
    from image_description.cli.social_post_builder_cli import main as social_main
    folder = tmp_path / 'images'
    add_image(folder, 'a.jpg')
    main([str(folder)])
    args = ['images', '--image-root', str(tmp_path)] if use_root else [str(folder)]
    social_main(args)
    assert (folder / 'a.social.json').exists()
    assert not (folder / 'gallery.social.json').exists()
