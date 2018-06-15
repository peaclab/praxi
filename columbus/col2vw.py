import ast
import pickle
next_id = 1
with open("label_ids.p", "rb") as f:
    label_ids = pickle.load(f)
with open("ks_tr_vw.in", "a") as outf:
    with open("result", "r") as f:
        for ln in f:
            if '+' in ln:
                ln = ln.strip()
                if not ln:
                    continue
                split_str = ln.split('\t', 1)
                labels = split_str[0].split('+')
                raw_tags = ast.literal_eval(split_str[1].strip())
                tags = []
                for tag in raw_tags:
                    tags.append(tag.replace(":", "_"))
                for label in labels:
                    if label not in label_ids:
                        label_ids[label] = next_id
                        next_id += 1
                    outf.write(str(label_ids[label]) + ":1.0 ")
                outf.write("{name}| {tags}\n".format(name=split_str[0].strip(),
                                                     tags=" ".join(tags)))
with open("multi_label_ids.p", "w") as f:
    pickle.dump(label_ids, f)
