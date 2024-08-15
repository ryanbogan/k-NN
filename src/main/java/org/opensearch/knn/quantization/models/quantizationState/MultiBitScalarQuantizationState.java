/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.io.IOException;

/**
 * MultiBitScalarQuantizationState represents the state of multi-bit scalar quantization,
 * including the thresholds used for quantization.
 */
@Getter
@NoArgsConstructor // No-argument constructor for deserialization
@AllArgsConstructor
public final class MultiBitScalarQuantizationState implements QuantizationState {
    private ScalarQuantizationParams quantizationParams;
    /**
     * The threshold values for multi-bit quantization, organized as a 2D array
     * where each row corresponds to a different bit level.
     *
     * For example:
     * - For 2-bit quantization:
     *   thresholds[0] {0.5f, 1.5f, 2.5f}  // Thresholds for the first bit level
     *   thresholds[1] {1.0f, 2.0f, 3.0f}  // Thresholds for the second bit level
     * - For 4-bit quantization:
     *   thresholds[0] {0.1f, 0.2f, 0.3f}
     *   thresholds[1] {0.4f, 0.5f, 0.6f}
     *   thresholds[2] {0.7f, 0.8f, 0.9f}
     *   thresholds[3] {1.0f, 1.1f, 1.2f}
     *
     * Each column represents the threshold for a specific dimension in the vector space.
     */
    private float[][] thresholds;

    @Override
    public ScalarQuantizationParams getQuantizationParams() {
        return quantizationParams;
    }

    /**
     * This method is responsible for writing the state of the MultiBitScalarQuantizationState object to an external output.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     *
     * @param out the StreamOutput to write the object to.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(Version.CURRENT.id); // Write the version
        quantizationParams.writeTo(out);
        out.writeVInt(thresholds.length); // Number of rows
        for (float[] row : thresholds) {
            out.writeFloatArray(row); // Write each row as a float array
        }
    }

    /**
     * This method is responsible for reading the state of the MultiBitScalarQuantizationState object from an external input.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     *
     * @param in the StreamInput to read the object from.
     * @throws IOException if an I/O error occurs during deserialization.
     */
    public MultiBitScalarQuantizationState(StreamInput in) throws IOException {
        int version = in.readVInt(); // Read the version
        this.quantizationParams = new ScalarQuantizationParams(in, version);
        int rows = in.readVInt(); // Read the number of rows
        this.thresholds = new float[rows][];
        for (int i = 0; i < rows; i++) {
            this.thresholds[i] = in.readFloatArray(); // Read each row as a float array
        }
    }

    /**
     * Serializes the current state of this MultiBitScalarQuantizationState object into a byte array.
     * This method uses the QuantizationStateSerializer to handle the serialization process.
     *
     * <p>The serialized byte array includes all necessary state information, such as the thresholds
     * and quantization parameters, ensuring that the object can be fully reconstructed from the byte array.</p>
     *
     * <pre>
     * {@code
     * MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);
     * byte[] serializedState = state.toByteArray();
     * }
     * </pre>
     *
     * @return a byte array representing the serialized state of this object.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public byte[] toByteArray() throws IOException {
        return QuantizationStateSerializer.serialize(this);
    }

    /**
     * Deserializes a MultiBitScalarQuantizationState object from a byte array.
     * This method uses the QuantizationStateSerializer to handle the deserialization process.
     *
     * <p>The byte array should contain serialized state information, including the thresholds
     * and quantization parameters, which are necessary to reconstruct the MultiBitScalarQuantizationState object.</p>
     *
     * <pre>
     * {@code
     * byte[] serializedState = ...; // obtain the byte array from some source
     * MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.fromByteArray(serializedState);
     * }
     * </pre>
     *
     * @param bytes the byte array containing the serialized state.
     * @return the deserialized MultiBitScalarQuantizationState object.
     * @throws IOException if an I/O error occurs during deserialization.
     */
    public static MultiBitScalarQuantizationState fromByteArray(final byte[] bytes) throws IOException {
        return (MultiBitScalarQuantizationState) QuantizationStateSerializer.deserialize(bytes, MultiBitScalarQuantizationState::new);
    }
}