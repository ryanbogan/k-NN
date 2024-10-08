/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import java.io.IOException;

public interface KNNIterator {
    int nextDoc() throws IOException;

    float score();
}
