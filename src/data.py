from pathlib import Path
import polars as pl


class Data:

    def __init__(self, path: str):
        self.path = Path(str(path))
        self.run()

    def _load(self) -> dict:
        return dict(
                (i.name.split(".")[0], pl.read_parquet(str(i)))
                for i in self.path.rglob("*.parquet")
            )


    @staticmethod
    def _tidy(data: dict):
        data.update(
            order_products=pl.concat(
                [
                    data.pop('order_products__prior').with_columns(
                        pl.lit('prior').alias('eval_set')
                    ),
                    data.pop('order_products__train').with_columns(
                        pl.lit('train').alias('eval_set')
                    ),
                ]
            )
        )
        return data

    def run(self): 
        data = self._tidy(data=self._load())
        if data:
            for k, v in data.items():
                setattr(self, k, v)

