#!/bin/zsh

for ipynb in **/*.ipynb
do
    jupyter nbconvert --to markdown $ipynb
done

