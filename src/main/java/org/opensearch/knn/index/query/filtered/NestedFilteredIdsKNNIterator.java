/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to score. However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedFilteredIdsKNNIterator extends FilteredIdsKNNIterator {
    private final DocIdSetIterator parentDocIdSetIterator;

    NestedFilteredIdsKNNIterator(
        final DocIdSetIterator filterIdsArray,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final DocIdSetIterator parentDocIdSetIterator
    ) throws IOException {
        this(filterIdsArray, queryVector, knnFloatVectorValues, spaceType, parentDocIdSetIterator, null, null);
    }

    public NestedFilteredIdsKNNIterator(
        final DocIdSetIterator filterIdsArray,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final DocIdSetIterator parentDocIdSetIterator,
        final byte[] quantizedVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) throws IOException {
        super(filterIdsArray, queryVector, knnFloatVectorValues, spaceType, quantizedVector, segmentLevelQuantizationInfo);
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
            knnFloatVectorValues.advance(docId);
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
