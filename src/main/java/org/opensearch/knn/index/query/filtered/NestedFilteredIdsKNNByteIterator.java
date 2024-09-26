/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to score. However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedFilteredIdsKNNByteIterator extends FilteredIdsKNNByteIterator {
    private final DocIdSetIterator parentDocIdSetIterator;

    public NestedFilteredIdsKNNByteIterator(
        final DocIdSetIterator filterIdsArray,
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType,
        final DocIdSetIterator parentDocIdSetIterator
    ) throws IOException {
        super(filterIdsArray, queryVector, binaryVectorValues, spaceType);
        this.parentDocIdSetIterator = parentDocIdSetIterator;
    }

    /**
     * Advance to the next best child doc per parent and update score with the best score among child docs from the parent.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next best child doc id
     */
    @Override
    public int nextDoc() throws IOException {
        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }

        currentScore = Float.NEGATIVE_INFINITY;
        int currentParent = parentDocIdSetIterator.nextDoc();
        int bestChild = -1;

        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
            binaryVectorValues.advance(docId);
            float score = computeScore();
            if (score > currentScore) {
                bestChild = docId;
                currentScore = score;
            }
            docId = docIdSetIterator.nextDoc();
        }

        return bestChild;
    }
}
