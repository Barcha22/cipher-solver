import re
from pathlib import Path

ALPHABET = set("abcdefghijklmnopqrstuvwxyz ")
LENGTHS = [50,100,200,500,1000]
N = 40

class Data:
    def __init__(self):
        self.TRAIN_BOOKS = {"austen-emma.txt", "austen-persuasion.txt", "austen-sense.txt",
                       "chesterton-ball.txt", "chesterton-thursday.txt"}
        self.TEST_BOOKS  = {"melville-moby_dick.txt", "shakespeare-hamlet.txt"}

    # function to clean text keeping only a-z characters
    def clean_text(self,raw: str)->str:
        text = raw.lower() #everythign lowercase
        text = text.replace("'", "").replace("`", "").replace("\u2019", "").replace("\u2018", "") #removing apostrophes
        text = re.sub(r"[^a-z]+"," ",text) #anything that is not a-z is substituted with " "
        text = re.sub(r"\s+"," ",text).strip() #finds all the whitespaces and replaces with " "
        return text

    # function to read all files from raw folder and then clean them with clean_text function, 5 for training 2 for testing
    def process_all_books(self):
        root_dir = Path(__file__).resolve().parent.parent
        raw_data_path = root_dir/"data/raw"
        clean_data_path = root_dir/"data/clean"
        test_data_path = root_dir/"data/test"

        clean_data_path.mkdir(parents=True, exist_ok=True)
        test_data_path.mkdir(parents=True, exist_ok=True)

        clean_files = {p.name for p in clean_data_path.iterdir() if p.is_file()}
        test_files = {p.name for p in test_data_path.iterdir() if p.is_file()}

        if clean_files != self.TRAIN_BOOKS or test_files != self.TEST_BOOKS:
            for item in raw_data_path.iterdir():
                if not item.is_file():
                    continue
                try:
                    content = item.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    content = item.read_text(encoding='cp1252',errors='replace')
    
                cleaned_text = self.clean_text(content)
                
                #stats
                raw_len = len(content)
                clean_len = len(cleaned_text)
                pct_kept = (clean_len / raw_len * 100) if raw_len > 0 else 0.0
                distinct_symbols = sorted(set(content))
                print("----------STATS--------------")
                print(
                    f"Book: {item.stem} | "
                    f"Raw chars: {raw_len:,} | "
                    f"Clean chars: {clean_len:,} | "
                    f"Kept: {pct_kept:.2f}% | "
                    f"Distinct symbols ({len(distinct_symbols)}): {''.join(distinct_symbols)!r}"
                )
    
                output_path = test_data_path/item.name if item.name in self.TEST_BOOKS else clean_data_path/item.name

                with open(output_path,'w',encoding='utf-8') as f:
                    f.write(cleaned_text)
            print("All files processed.")
        else:
             print("Files already processed.")

    # function to check wether the text corpus is clean or not in the folders
    def check_files(self):
        root_dir = Path(__file__).resolve().parent.parent
        clean_dir = root_dir / "data/clean"
        test_dir  = root_dir / "data/test"

        clean_files = {p.name for p in clean_dir.iterdir() if p.is_file()}
        test_files  = {p.name for p in test_dir.iterdir()  if p.is_file()}
        print(f"data/clean: {len(clean_files)} files, data/test: {len(test_files)} files")
        all_ok = True

        if clean_files != self.TRAIN_BOOKS:
            all_ok = False
            print(f"  MISMATCH clean/: extra={clean_files - self.TRAIN_BOOKS} "
                  f"missing={self.TRAIN_BOOKS - clean_files}")
        if test_files != self.TEST_BOOKS:
            all_ok = False
            print(f"  MISMATCH test/: extra={test_files - self.TEST_BOOKS} "
                  f"missing={self.TEST_BOOKS - test_files}")

        for directory, files in ((clean_dir, clean_files), (test_dir, test_files)):
            for name in sorted(files):
                text = (directory / name).read_text(encoding="utf-8")
                chars   = set(text)
                extras  = chars - ALPHABET      
                missing = ALPHABET - chars           
                status = "OK" if not extras else f"FAIL extras={''.join(sorted(extras))!r}"
                note = f" (missing: {''.join(sorted(missing))})" if missing else ""
                print(f"{name:28s} {status}{note} | {len(text):,} chars")
                if extras:
                    all_ok = False

        print("\nRESULT:", "ALL OK" if all_ok else "PROBLEMS FOUND")
        return all_ok

    def build_passages(self):
        test_dir = Path(__file__).resolve().parent.parent / "data/test"
        out_dir = test_dir/"passages"
        out_dir.mkdir(exist_ok=True)

        for book in self.TEST_BOOKS:
            text = (test_dir/book).read_text(encoding='utf-8')
            for L in LENGTHS:
                pieces = [text[i * (len(text) - L) // N : i * (len(text) - L) // N + L] for i in range(N)]
                out = out_dir / f"{Path(book).stem}_{L}.txt"
                out.write_text("\n".join(pieces), encoding="utf-8")
                print(f"{out.name}: {len(pieces)} pieces")

if __name__ == "__main__":
    data = Data()
    data.process_all_books()
    data.check_files()
    data.build_passages()


             
        
