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


## Labeled LDA

1. Install JGibbLabeledLDA module

Please follow the instructions in `JGibbLabeledLDA.md`.

2. Preprocess the data input

```
# Generate training data file.
python3 JLabeledLDA.py -o labeled.lda -a aspect.out -p polarity.out -t test.out

# Generate testing data file.
python3 JLabeledLDA.py -o test.lda -t test.out
gzip labeled.out
```

* It will replace the `labeled.lda` by `labeled.lda.gz`.

3. Train model

```
java -cp bin:lib/args4j-2.0.6.jar:lib/trove-3.0.3.jar jgibblda.LDA -est -ntopics 5 -dir ./model -dfile labeled.out.gz -model <model> -niters 300
```

4. Inference answer by model

```
java -cp bin:lib/args4j-2.0.6.jar:lib/trove-3.0.3.jar jgibblda.LDA -inf -dir ./model -model <model> -niters 50 -dfile labeled.out.gz
```

5. Generate answer!

```
gunzip <file>.<model>.theta.gz
python3 After_JGibbLabeledLDA.py -r <file>.<model>.theta -t test.out -q test.csv -o <answer.csv> [-d debug.out] [-x 0.3]
```

Please use `python3 After_JGibbLabeledLDA.py -h` for further detailed usage.


## Polarity Decider

### Usage (train)

```bash
$ python3 polarity_decider.py [-p polarity_review.out] [-d NTUSD_path] [--train]
```

* `[-p polarity_review.out]` put the corresponding path here
* `[-d NTUSD_path]` put the path that NTUSD_pos.txt and NTUSD_neg.txt are
* `[--train]` specifies the program to run training

### Usage (test per review)

```bash
$ python3 polarity_decider.py [-t test_review.out] [-q test.csv] [-o output_file] [--test]
```

* `[-t test_review.out]` put the corresponding path here
* `[-q test.csv]` put the corresponding path here
* `[-o output file]` put output file name here
* `[--test]` specifies the program to run testing

### Usage (test per sentence)

```bash
$ python3 polarity_decider.py [-t test_review.out] [-o output_file] [--test] [-s]
```

* `[-t test_review.out]` put the corresponding path here (Remember to put the per-sentence version)
* `[-o output file]` put output file name here
* `[--test]` specifies the program to run testing
* `[-s]` specifies the program to run testing per sentence


## Merge Result

### Usage (per review)

```bash
$ python3 merge_results.py [-a aspect_result.csv] [-p polarity_result.csv]
```

* `[-a aspect]` put the result file that only uses aspect
* `[-p polarity_result.csv]` put the result file that only uses polarity

### Usage (per sentence)

```bash
$ python3 merge_results.py [-a aspect_result.csv] [-p polarity_result.csv] [-q test.csv] [-s]
```

* `[-a aspect]` put the result file that only uses aspect
* `[-p polarity_result.csv]` put the result file that only uses polarity
* `[-q test.csv]` put the corresponding path here.
* `[-s]` specifies the program to merge per-sentence results


## Aspect decider

### Notes: Parse files before
```bash
python3 parser.py -a -n -i data/aspect_review.txt -o data/aspect.out
python3 parser.py -t -n -i  data/test_review.txt -o data/test.out
python3 parser.py -p -n -i data/polarity_review.txt -o polarity.out

python3 simplifier.py -i data/aspect.out -o data/aspect_simpl.out
python3 simplifier.py -i data/test.out -o data/test_simpl.out
python3 simplifier.py -i data/polarity.out -o data/polarity_simpl.out
```

### Usage

```bash
python3 aspect_decider.py -p data/polarity_simpl.out -a data/aspect_simpl.out -t data/test_simpl.out -q data/test.csv -o Aspect_per_sentence.csv -d 0.57
```
