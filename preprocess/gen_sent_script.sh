#!/bin/sh

python generate_sentences.py 0 &
python generate_sentences.py 200 &
python generate_sentences.py 400 &
python generate_sentences.py 600 &
python generate_sentences.py 800 &
python generate_sentences.py 1000 &
python generate_sentences.py 1200 &
python generate_sentences.py 1400 &
python generate_sentences.py 1600 &
python generate_sentences.py 1800 &
python generate_sentences.py 2000 &
python generate_sentences.py 2200 &
python generate_sentences.py 2400 &
python generate_sentences.py 2600 &
python generate_sentences.py 2800 &

wait
echo "Preprocessing complete"
