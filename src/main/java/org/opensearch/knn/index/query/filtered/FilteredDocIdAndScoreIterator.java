/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.util.Bits;

import java.io.IOException;

public class FilteredDocIdAndScoreIterator extends FilteredDocIdSetIterator {

    private final Bits liveDocs;

    /**
     * Constructor
     *
     * @param innerIterator inner iterator
     */
    public FilteredDocIdAndScoreIterator(DocIdAndScoreIterator innerIterator, Bits liveDocs) {
        super(innerIterator);
        this.liveDocs = liveDocs;
    }

    @Override
    protected boolean match(int doc) throws IOException {
        return liveDocs == null || liveDocs.get(doc);
    }

    public long[] getDocs() {
        return ((DocIdAndScoreIterator) _innerIter).getDocs();
    }
}
