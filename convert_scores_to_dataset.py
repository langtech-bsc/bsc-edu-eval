import sys, os
import json
import numpy as np
from datasets import load_dataset, DatasetDict, Value, Dataset
from collections import Counter


if __name__ == "__main__":
    num_paths = int(sys.argv[1]) # 1
    paths = [sys.argv[i+2] for i in range(num_paths)] # /path/to/the/dataset
    num_ds_paths = int(sys.argv[num_paths+2]) # 1
    ds_paths = [sys.argv[num_paths+3+i] for i in range(num_ds_paths)] # /path/to/the/jsonl/with/examples/of/description/with/intent/to/sell
    model = sys.argv[num_paths+num_ds_paths+3] # /model/used/to/predict/scores
    out_path = sys.argv[num_paths+num_ds_paths+4] # /output/path
    do_mix_dataset_with_ds_category = bool(int(sys.argv[num_paths+num_ds_paths+5])) # 0
    
    
    do_get_ds_category_separately = False
    
    test_size = 2000
    ds_to_take = 45000
    ds_to_take_for_test = 1000
    
    js_ds_train = []
    js_ds_test = []
    if do_mix_dataset_with_ds_category or do_get_ds_category_separately:
        ds_to_take = 45000
        for ds_path in ds_paths:
            js_ds = [list(map(lambda x: (x["text"], 0), [json.loads(line)]))[0] for line in open(ds_path, "r").read().splitlines()[:ds_to_take+ds_to_take_for_test]]
            js_ds_train.extend(js_ds[:ds_to_take])
            js_ds_test.extend(js_ds[ds_to_take:])
            print(f"File {ds_path} has been read.")
    
    if do_get_ds_category_separately:
        np.random.shuffle(js_ds_train)
        js_ds_train_unzipped = list(zip(*js_ds_train))
        js_ds_test_unzipped = list(zip(*js_ds_test))
        dataset = DatasetDict({"train": Dataset.from_dict({'text': js_ds_train_unzipped[0], 'label': js_ds_train_unzipped[1]}),
                           "test": Dataset.from_dict({'text': js_ds_test_unzipped[0], 'label': js_ds_test_unzipped[1]})
                           })
        dataset.save_to_disk(out_path)
    else:
        js_train = []
        js_test = []
        for path in paths:
            js = []
            filename = f"{model}_responses.jsonl"
            js.extend([list(map(lambda x: (x["dataset_instance"]["text"], int(x["parsed_response"]["score"]) if x["parsed_response"]["score"] else -1), [json.loads(line)]))[0] for line in open(os.path.join(path, filename), "r").read().splitlines()])
            print(f"File {filename} has been read.")
            for i in range(50000, 450001, 50000):
                filename = f"{model}_responses_{i}_{i+50000}.jsonl"
                js.extend([list(map(lambda x: (x["dataset_instance"]["text"], int(x["parsed_response"]["score"]) if x["parsed_response"]["score"] else -1), [json.loads(line)]))[0] for line in open(os.path.join(path, filename), "r").read().splitlines()])
                print(f"File {filename} has been read.")
            train_size = len(js) - test_size
            js_train.extend([ex for ex in js[:train_size] if 0<=ex[1]])
            js_test.extend([ex for ex in js[train_size:] if 0<=ex[1]])
            
        js_train.extend(js_ds_train)
        np.random.shuffle(js_train)
        js_train_shuffled_unzipped = list(zip(*js_train))
        js_test_unzipped = list(zip(*js_test))
        print(Counter(js_train_shuffled_unzipped[1]))
        print(Counter(js_test_unzipped[1]))
        dataset = DatasetDict({"train": Dataset.from_dict({'text': js_train_shuffled_unzipped[0], 'label': js_train_shuffled_unzipped[1]}),
                           "test": Dataset.from_dict({'text': js_test_unzipped[0], 'label': js_test_unzipped[1]})
                           })
        dataset.save_to_disk(out_path)
    