from logging import getLogger
from datasets import Dataset
from contextlib import contextmanager


class NoOpAccelerator:
    """Dummy accelerator for non-distributed / no-accelerator scenarios."""

    @contextmanager
    def main_process_first(self):
        yield


class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config.get("accelerator", NoOpAccelerator())
        self.logger = getLogger()

        self.all_seqs = {}
        self.id_mapping = {"user2id": {"[PAD]": 0}, "item2id": {"[PAD]": 0}, "id2user": ["[PAD]"], "id2item": ["[PAD]"]}
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return (
            f"[Dataset] {self.__class__.__name__}\n"
            f"\tNumber of users: {self.n_users}\n"
            f"\tNumber of items: {self.n_items}\n"
            f"\tNumber of interactions: {self.n_interactions}\n"
            f"\tAverage item sequence length: {self.avg_item_seq_len}"
        )

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for user in self.all_seqs:
            n_inters += len(self.all_seqs[user]["item"])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping["user2id"]

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping["item2id"]

    def _download_and_process_raw(self):
        """
        This method should be implemented in the subclass.
        It is responsible for downloading and processing the raw data.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def _leave_one_out(self):
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        keys = ["user"] + list(map(lambda x: x + "_seq", next(iter(self.all_seqs.values())).keys()))
        datasets = {
            "train": {k: [] for k in keys},
            "val": {k: [] for k in keys},
            "test": {k: [] for k in keys},
        }
        for user in self.all_seqs:
            datasets["test"]["user"].append(user)
            for k, seq in self.all_seqs[user].items():
                datasets["test"][f"{k}_seq"].append(seq)
            if len(next(iter(self.all_seqs[user].values()))) > 1:
                datasets["val"]["user"].append(user)
                for k, seq in self.all_seqs[user].items():
                    datasets["val"][f"{k}_seq"].append(seq[:-1])
            if len(next(iter(self.all_seqs[user].values()))) > 2:
                datasets["train"]["user"].append(user)
                for k, seq in self.all_seqs[user].items():
                    datasets["train"][f"{k}_seq"].append(seq[:-2])
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def prepare_temporal_stage(self, reviews, stage="base"):
        """Explicit raw-review adapter; does not reuse legacy processed caches.

        Returns existing HF Dataset splits and stores raw diagnostic metadata
        separately. Model-ready SID/catalog encoding is a later stage.
        """
        import copy
        from genrec.temporal import build_temporal_protocol

        if stage not in ("base", "update"):
            raise ValueError("stage must be base or update")
        if not self.config.get("explicit_causal_examples", False):
            raise ValueError("Set explicit_causal_examples=True before temporal preparation")
        if self.config.get("n_user_tokens", 0) != 0:
            raise ValueError("Temporal mode currently requires n_user_tokens=0 for unseen users")
        protocol = build_temporal_protocol(
            reviews,
            max_history=self.config["max_item_seq_len"],
            duplicate_policy=self.config.get("duplicate_policy", "drop_exact"),
        )
        self.temporal_protocol = protocol
        self.id_mapping = copy.deepcopy(protocol["registries"][stage])
        names = {"train": f"{stage}_train", "val": f"{stage}_val"}
        if stage == "update":
            names["test"] = "future_test"
        self.temporal_metadata = {}
        splits = {}
        for split, name in names.items():
            raw = protocol["examples"][name]
            self.temporal_metadata[split] = copy.deepcopy(raw)
            splits[split] = Dataset.from_list([
                {"user": row["user"], "item_seq": row["item_seq"]}
                for row in raw
            ])
        self.split_data = splits
        return splits

    def split(self):
        """
        Split the dataset into train, validation, and test sets based on the specified split strategy.

        Returns:
            datasets (dict): A dictionary containing the train and test datasets.
        """
        if self.split_data is not None:
            return self.split_data

        if self.config.get("test_nrows") is not None:
            self.all_seqs = {
                k: self.all_seqs[k] for k in list(self.all_seqs.keys())[: self.config["test_nrows"]]
            }

        split_strategy = self.config["split"]
        if split_strategy in ["leave_one_out", "last_out"]:
            datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f"Split strategy [{split_strategy}] not implemented.")

        self.split_data = datasets
        return self.split_data

    def log(self, message, level="info"):
        from genrec.utils import log

        return log(message, self.config.get("accelerator"), self.logger, level=level)
