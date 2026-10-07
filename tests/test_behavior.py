import importlib.util
from pathlib import Path
from utils.chunking import chunk_text, enrich_metadata

def test_chunking_preserves_overlap_and_tail():
    assert chunk_text('a b c d e f g', max_tokens=3, overlap=1) == ['a b c', 'c d e', 'e f g', 'g']
    assert chunk_text(' \n ') == []
    assert chunk_text('one two', max_tokens=3, overlap=0) == ['one two']

def test_metadata_does_not_mutate_extra():
    extra = {'page': 2}
    result = enrich_metadata('doc', 'text', 'file.txt', extra)
    assert result == {'doc_id':'doc', 'modality':'text', 'source':'file.txt', 'page':2}
    result['page'] = 3
    assert extra == {'page':2}

def test_registration_and_password_verification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[1] / 'utils/auth.py'
    spec = importlib.util.spec_from_file_location('isolated_auth', path)
    auth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auth)
    assert auth.register_user(' Alex ', 'test-password')[0] is True
    assert auth.register_user('alex', 'other')[0] is False
    assert auth.authenticate_user('ALEX', 'test-password')[0] is True
    assert auth.authenticate_user('alex', 'wrong')[0] is False
    assert auth.authenticate_user('missing', 'test-password')[0] is False
    assert auth.register_user('', 'test-password')[0] is False
    assert 'test-password' not in auth.USERS_PATH.read_text()
