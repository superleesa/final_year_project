from abc import ABC
import pandas as pd


class Tracker(ABC):
    def __init__(self, dump_path: str) -> None:
        self.dump_path = dump_path
        self.records = []
    
    def add_record(self, record: float) -> None:
        self.records.append(record)
    
    def dump(self) -> None:
        ...


class SingelValueTracker(Tracker):
    def __init__(self, dump_path: str, record_name: str) -> None:
        super().__init__(dump_path)
        self.record_name = record_name
    
    def dump(self) -> None:
        df = pd.DataFrame({self.record_name: self.records})
        df.to_csv(self.dump_path)