/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import lombok.Getter;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

@Getter
public class DocIdAndScoreIterator extends DocIdSetIterator {

    private final int[] docs;
    private final float[] scores;
    private int doc = -1;

    public DocIdAndScoreIterator(int[] docs, float[] scores) {
        this.docs = docs;
        this.scores = scores;
    }

    @Override
    public int docID() {
        return doc;
    }

    @Override
    public int nextDoc() throws IOException {
        return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
        doc = target;
        if (doc > docs.length) {
            doc = NO_MORE_DOCS;
        }
        return doc;
    }

    @Override
    public long cost() {
        return docs.length + 1;
    }

    public long cardinality() {
        return docs.length;
    }
}
