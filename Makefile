CC=python
EXEC=src/troll_identifier.py

DEV_LARGE=data/large-csv/dev-best200.csv
DEV_MEDIUM=data/medium-csv/dev-best50.csv
DEV_SMALL=data/small-csv/dev-best10.csv

TRAIN_LARGE=data/large-csv/train-best200.csv
TRAIN_MEDIUM=data/medium-csv/train-best50.csv
TRAIN_SMALL=data/small-csv/train-best10.csv

TEST_LARGE=data/large-csv/test-best200.csv
TEST_MEDIUM=data/medium-csv/test-best50.csv
TEST_SMALL=data/small-csv/test-best10.csv

all:
	${CC} ${EXEC}

large:
	${CC} ${EXEC} ${TRAIN_LARGE} ${DEV_LARGE}

medium:
	${CC} ${EXEC} ${TRAIN_MEDIUM} ${DEV_MEDIUM}

small:
	${CC} ${EXEC} ${TRAIN_SMALL} ${DEV_SMALL}

test:
	${CC} ${EXEC} ${TRAIN_SMALL} ${TEST_SMALL}
