/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

@Log4j2
public class QuantizationStateCache {

    private final ConcurrentHashMap<String, QuantizationState> cache = new ConcurrentHashMap<>();
    private final Lock lock = new ReentrantLock();
    private static QuantizationStateCache instance;

    private QuantizationStateCache() {}

    public static QuantizationStateCache getInstance() {
        if (instance == null) {
            instance = new QuantizationStateCache();
        }
        return instance;
    }

    public QuantizationState getQuantizationState(String fieldName) {
        return cache.get(fieldName);
    }

    public void addQuantizationState(String fieldName, QuantizationState quantizationState) {
        lock.lock();
        try {
            cache.put(fieldName, quantizationState);
        } finally {
            lock.unlock();
        }
    }

    public void evict(String fieldName) {
        lock.lock();
        try {
            cache.remove(fieldName);
        } finally {
            lock.unlock();
        }
    }

    public void clear() {
        cache.clear();
    }
}