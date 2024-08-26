import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import typer
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_show_locals=False)


class EvalDataset(Dataset):
    def __init__(self, dir: Path, prompt_path: Path, img_preproc) -> None:
        super().__init__()
        self.img_paths = natsorted(dir.rglob("*.png"))
        with open(prompt_path) as f:
            data = json.load(f)["labels"]
            self.prompts = [x[1] for x in data]
            self.lookup = {x[0]: x[1] for x in data}
        self.img_preproc = img_preproc
        self.dir = dir
        assert len(self.img_paths) == 30000

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            prompt_idx = int(img_path.stem)
            prompt = self.prompts[prompt_idx]
        except Exception:
            prompt = self.lookup[str(img_path.relative_to(self.dir))]
        prompt = "A photo depicts " + prompt
        image = self.img_preproc(Image.open(img_path.as_posix()))
        text = open_clip.tokenize([prompt])[0]

        return image, text


@app.command()
def main(
    dir: Path = typer.Argument(..., help="Sampled path", dir_okay=True, exists=True),
    prompt_path: Path = typer.Argument("dataset.json", help="prompt", file_okay=True, exists=True),
    outpath: Path = typer.Option("metrics/clip30k.txt", help="outpath", file_okay=True),
    batch_size: int = typer.Option(32, help="batch size"),
    use_dataloader: bool = typer.Option(False, help="Use dataloader"),
):
    device = "cuda"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s12b_b42k")
    model.float()
    model, _ = model.to(device), open_clip.get_tokenizer("ViT-g-14")

    img_paths = natsorted(dir.rglob("*.png"))
    mean_score = 0

    assert len(img_paths) == 30000
    if not use_dataloader:
        with open(prompt_path) as f:
            data = json.load(f)["labels"]
            prompts = [x[1] for x in data]

        for img_path in tqdm(img_paths):
            idx = int(img_path.stem)
            caption = prompts[idx]
            caption = "A photo depicts " + caption
            image = preprocess(Image.open(img_path.as_posix())).unsqueeze(0).to(device)
            text = open_clip.tokenize([caption]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                probs = F.cosine_similarity(image_features, text_features, dim=1)
                mean_score += probs[0]
        mean_score /= len(img_paths)
    else:
        dataset = EvalDataset(dir, prompt_path, preprocess)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        cos_sims = []
        with torch.no_grad():
            for images, texts in tqdm(dataloader):
                images_features = model.encode_image(images.to(device))
                texts_features = model.encode_text(texts.to(device))
                similarities = F.cosine_similarity(images_features, texts_features, dim=1)

                cos_sims.append(similarities)
        clip_score = torch.cat(cos_sims, dim=0).mean()
        mean_score = clip_score.detach().cpu()

    print("CLIP score:", mean_score)
    outpath.parent.mkdir(exist_ok=True, parents=True)
    with open(outpath, "a") as f:
        f.write(f"{dir}\t{mean_score.item()}\n")


if __name__ == "__main__":
    app()
