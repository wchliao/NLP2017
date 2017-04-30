# JGibb Labeled LDA


## Get it work!

```
git clone https://github.com/myleott/JGibbLabeledLDA
cd JGibbLabeledLDA/
mkdir bin
javac -d bin -cp ./lib/args4j-2.0.6.jar:./lib/trove-3.0.3.jar src/jgibblda/*
```

## Notes

1. It's file format must be Gzipped!
```
gzip <file>  # This will replaced the original file!
```

## Execution

### Pre processing

```
python3 JLabeledLDA.py -o <combined.labeled> -a aspect.out -p polarity.out -t test.out
```

For clearer usage explanation, please refer to `python3 JLabeledLDA.py -h`.

### Training

```
java -cp bin:lib/args4j-2.0.6.jar:lib/trove-3.0.3.jar jgibblda.LDA -est [-ntopics 5] -dir ./model -dfile labeled.out.gz -model <name> -niters 300
```

* For more parameter setting, please [go to here](http://jgibblda.sourceforge.net/).
* Remember, `labeled.out.gz` actually means `./model/labeled.out.gz`.
* Also, `-model <name>` is needed for it to save the model.

### Testing (Inference)

```
java -cp bin:lib/args4j-2.0.6.jar:lib/trove-3.0.3.jar jgibblda.LDA -inf -dir ./model -model <name> -niters 50 -dfile labeled.out.gz
```


