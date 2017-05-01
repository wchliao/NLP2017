#!/bin/bash

DATA_DIR=""
MODEL_NAME=""
ITERATION="300"
THRESHOLD="0.2"
LIB_DIR="JGibbLabeledLDA/"

while getopts "l:d:m:i:t:v" optname
  do
    case "$optname" in
      "l")
        LIB_DIR=$OPTARG
        ;;
      "d")
		DATA_DIR=$OPTARG
        echo "DATA_DIR is $DATA_DIR"
        ;;
      "m")
		MODEL_NAME=$OPTARG
        echo "MODEL_NAME is $MODEL_NAME"
        ;;
      "i")
		ITERATION=$OPTARG
        echo "ITERATION is $ITERATION"
        ;;
      "t")
		THRESHOLD=$OPTARG
        echo "THRESHOLD is $THRESHOLD"
        ;;
      "v")
		DEBUG=true
        echo "DEBUG mode is on."
        ;;
      "?")
        echo "Unknown option $OPTARG"
        ;;
      ":")
        echo "No argument value for option $OPTARG"
        ;;
      *)
      # Should not occur
        echo "Unknown error while processing options"
        ;;
    esac
  done

[ "$DATA_DIR" == "" ] && echo "You forget to set DATA_DIR!" && exit 1
[ "$MODEL_NAME" == "" ] && echo "You forget to set MODEL_NAME!" && exit 1

for file in $DATA_DIR/aspect.out $DATA_DIR/polarity.out $DATA_DIR/test.out $DATA_DIR/test.csv
do
    [ ! -f $file ] && echo "Please have $file exist!" && exit 1
done

####################################################

# Generate training data file.
echo "Generate LDA file input..."
python3 JLabeledLDA.py -o $DATA_DIR/labeled.lda -a $DATA_DIR/aspect.out -p $DATA_DIR/polarity.out -t $DATA_DIR/test.out
gzip $DATA_DIR/labeled.lda

# Generate testing data file.
python3 JLabeledLDA.py -o $DATA_DIR/test.lda -t $DATA_DIR/test.out
gzip $DATA_DIR/test.lda

####################################################

echo "Now training Labeled LDA model..."
java -cp $LIB_DIR/bin:$LIB_DIR/lib/args4j-2.0.6.jar:$LIB_DIR/lib/trove-3.0.3.jar jgibblda.LDA -est \
    -ntopics 5 -dir $DATA_DIR -dfile labeled.lda.gz -model $MODEL_NAME -niters $ITERATION

echo "Now inferencing testing data..."
java -cp $LIB_DIR/bin:$LIB_DIR/lib/args4j-2.0.6.jar:$LIB_DIR/lib/trove-3.0.3.jar jgibblda.LDA -inf \
    -dir $DATA_DIR -dfile test.lda.gz -model $MODEL_NAME -niters $ITERATION

####################################################

gunzip $DATA_DIR/test.lda.$MODEL_NAME.theta.gz

[ $DEBUG ] && DEBUG_ARGU="-d $DATA_DIR/debug.out"

python3 After_JGibbLabeledLDA.py -r $DATA_DIR/test.lda.$MODEL_NAME.theta \
    -t $DATA_DIR/test.out -q $DATA_DIR/test.csv -o $DATA_DIR/lda_result.csv $DEBUG_ARGU -x $THRESHOLD

echo "Labeled LDA done!"
