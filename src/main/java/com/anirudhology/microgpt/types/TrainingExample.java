package com.anirudhology.microgpt.types;

/**
 * Represents one supervise training sample
 *
 * @param context fixed size window of previous tokens
 * @param target  the actual next token
 */
public record TrainingExample(int[] context, int target) {
}
