from pathlib import Path
from typing import List

import hpsv2
import typer

app = typer.Typer()


@app.command()
def main(
    input_dirs: List[Path] = typer.Argument(..., help="input image dir"),
):
    for input_dir in input_dirs:
        out = hpsv2.evaluate(input_dir.as_posix())
        print("HPS score:", out)


if __name__ == "__main__":
    app()
