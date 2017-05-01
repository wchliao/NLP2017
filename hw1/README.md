# Homework 1


## Parser

### Notes

`stopwords_tw.txt` and `dict.txt.big` can be extended to fit hotel comment domain,
and perhaps using SCPCD is a great option for find new words! (TODO)

### Usage

```
python3 parser.py -a -i data/aspect_review.txt -o aspect.out
python3 parser.py -p -i data/polarity_review.txt -o polarity.out
python3 parser.py -t -i data/test_review.txt -o test.out
```

For detailed usage, please use `python3 parser.py -h`


## Polarity Decider

### Usage

```bash
$ python3 polarity_decider.py [-p polarity_review.out] [-t test_review.out] [-q test.csv] [-o output file] [--train] [--test]
```

* `[-p polarity_review.out]`, `[-t test_review.out]` and `[-q test.csv]` put corresponding paths here
* `[-o output file]` put output file name here
* `[--train]` specifies the program to run training
* `[--test]` specifies the program to run testing


## Merge Result

### Usage

```bash
$ python3 merge_results.py [-a aspect_result.csv] [-p polarity_result.csv]
```

* `[-a aspect]` put the result file that only uses aspect
* `[-p polarity_result.csv]` put the result file that only uses polarity

## Aspect decider

### Notes: Parse files before
```bash
python3 parser.py -a -n -i data/aspect_review.txt -o data/aspect.out
python3 parser.py -t -n -i  data/test_review.txt -o data/test.out
python3 parser.py -p -n -i data/polarity_review.txt -o polarity.out

python3 simplifier.py -i data/aspect.out data/aspect_simpl.out
python3 simplifier.py -i data/test.out data/test_simpl.out
python3 simplifier.py -i data/polarity.out data/polarity_simpl.out
```

### Usage

```bash
python3 aspect_decider.py -p data/polarity_simpl.out -a data/aspect_simpl.out -t data/test_simpl.out -q data/test.csv -o Aspect_per_sentence.csv -d 0.44
```
