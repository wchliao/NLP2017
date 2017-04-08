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
