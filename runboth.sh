#!/bin/bash
./minimap2 -x ava-pb -t8 reads.fq reads.fq | gzip -1 > reads.paf.gz
./minimap2 -x ava-pb -J8 reads.fq reads.fq | gzip -1 > reads2.paf.gz