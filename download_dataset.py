import argparse
import os
import zipfile

import gdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://drive.google.com/uc?id=1aPHE00zkDhEV1waJKhaOJMdN6-lUc0iT",
    )
    parser.add_argument("--output", default="archivo.zip")
    parser.add_argument("--destino", default="datos_zip")
    args = parser.parse_args()

    gdown.download(args.url, args.output, quiet=False)

    os.makedirs(args.destino, exist_ok=True)
    with zipfile.ZipFile(args.output, "r") as zip_ref:
        zip_ref.extractall(args.destino)

    print(f"Extraido en: {os.path.abspath(args.destino)}")


if __name__ == "__main__":
    main()
