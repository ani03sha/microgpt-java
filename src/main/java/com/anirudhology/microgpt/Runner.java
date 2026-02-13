package com.anirudhology.microgpt;

import com.anirudhology.microgpt.tokenizer.CharTokenizer;
import com.anirudhology.microgpt.tokenizer.TextCorpus;

import java.util.List;

public class Runner {

    static void main() {
        TextCorpus corpus = new TextCorpus();
        List<String> docs = corpus.readCorpus("input.txt");

        CharTokenizer tok = new CharTokenizer();
        tok.buildVocabulary(docs);

        System.out.println("Total number of docs: " + docs.size());
        System.out.println("Vocabulary size: " + tok.getVocabularySize());
        System.out.println("BOS id: " + tok.getBOSId());

        String sample = docs.getFirst();
        List<Integer> ids = tok.encode(sample);
        String roundTrip = tok.decode(ids);
        System.out.println("Sample: " + sample);
        System.out.println("Round-trip OK: " + sample.equals(roundTrip));

        List<Integer> bosWrapped = tok.withBOSOnBothSides(sample);
        System.out.println("With BOS: " + bosWrapped);
    }
}
