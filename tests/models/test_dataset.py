import datetime as dt
from dataclasses import dataclass

from src.models.dataset import HealthExamDataset


@dataclass
class _ManifestRow:
    person_id: str
    exam_date: dt.date
    exam_id: str


class _ManifestSlice:
    def __init__(self, row: _ManifestRow):
        self.row = row

    def to_pydict(self):
        return {
            'person_id': [self.row.person_id],
            'ExamDate': [self.row.exam_date],
            'exam_id': [self.row.exam_id],
        }


class _FakeManifest:
    def __init__(self, rows):
        self.rows = rows

    def slice(self, idx, count):
        return _ManifestSlice(self.rows[idx])

    @property
    def num_rows(self):
        return len(self.rows)


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def as_py(self):
        return self._value


class _FakeTable:
    def __init__(self, value):
        self._value = value
        self.num_rows = 1

    def column(self, name):
        assert name == 'tests'
        return [_FakeScalar(self._value)]


class _FakeMcinfoDataset:
    def __init__(self, tests_value):
        self.tests_value = tests_value

    def to_table(self, filter=None, columns=None):
        return _FakeTable(self.tests_value)


def _make_dataset_stub(result_lookup, *, use_pretokenized):
    """Create a lightweight HealthExamDataset instance for unit tests."""
    dataset = object.__new__(HealthExamDataset)
    exam_date = dt.date(2020, 1, 1)

    dataset.split_name = "stub"
    dataset.use_result = True
    dataset.use_pretokenized_result = use_pretokenized
    dataset._result_lookup = result_lookup
    dataset.EMPTY_RESULT_DATA = {'input_ids': [], 'attention_mask': []}
    dataset._manifest = _FakeManifest([
        _ManifestRow(person_id="p1", exam_date=exam_date, exam_id="exam-1")
    ])
    dataset._mcinfo_ds = _FakeMcinfoDataset(tests_value=[{'code': 'A1', 'type': 'PQ', 'value_num': 1.0}])
    dataset.mcinfo_materialized_path = None
    dataset._demographics_dict = {}
    dataset.use_interview = False
    dataset._initialized = True

    return dataset


def test_pretokenized_missing_result_uses_empty_sequences():
    dataset = _make_dataset_stub(result_lookup={}, use_pretokenized=True)

    sample = dataset[0]

    assert sample['result_input_ids'] == []
    assert sample['result_attention_mask'] == []


def test_pretokenized_existing_result_passthrough():
    exam_date = "2020-01-01"
    lookup = {
        ("p1", exam_date): {
            'input_ids': [5, 6],
            'attention_mask': [1, 1]
        }
    }
    dataset = _make_dataset_stub(result_lookup=lookup, use_pretokenized=True)

    sample = dataset[0]

    assert sample['result_input_ids'] == [5, 6]
    assert sample['result_attention_mask'] == [1, 1]


def test_raw_result_missing_falls_back_to_empty_string():
    dataset = _make_dataset_stub(result_lookup={}, use_pretokenized=False)

    sample = dataset[0]

    assert 'result_text' in sample
    assert sample['result_text'] == ""
