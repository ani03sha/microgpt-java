package com.anirudhology.microgpt.model;

import com.anirudhology.microgpt.autograd.Value;
import com.anirudhology.microgpt.data.TrainingExample;
import com.anirudhology.microgpt.nn.Embedding;
import com.anirudhology.microgpt.nn.Linear;
import com.anirudhology.microgpt.nn.PositionalEmbedding;
import com.anirudhology.microgpt.nn.RMSNormalization;
import com.anirudhology.microgpt.nn.TransformerBlock;
import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Full GPT-style transformer language model
 * <p>
 * Architecture: tokens -> embeddings -> positional ->
 * N*TransformerBlock -> RMSNorm -> Output head
 */
public class GPTLanguageModel {

    private final int vocabularySize;
    private final int blockSize;
    private final int embeddingDimension;

    // Input embeddings
    private final Embedding tokenEmbedding;
    private final PositionalEmbedding positionalEmbedding;

    // Transformer blocks (the core!)
    private final List<TransformerBlock> blocks;

    // Normalization applied right after embedding combination
    private final RMSNormalization embeddingNormalization;

    // Final normalization before output
    private final RMSNormalization finalNormalization;

    // Output projection: embeddingDimension -> vocabularySize
    private final Linear outputHead;

    // All parameters for optimization
    private final List<Value> allParameters;

    private final Random random;

    public GPTLanguageModel(
            int vocabularySize,
            int blockSize,
            int embeddingDimension,
            int numHeads,
            int numberOfLayers,
            boolean useMultiHead,
            long seed
    ) {
        this.vocabularySize = vocabularySize;
        this.blockSize = blockSize;
        this.embeddingDimension = embeddingDimension;
        this.random = new Random(seed);

        // Input embeddings
        this.tokenEmbedding = new Embedding(vocabularySize, embeddingDimension, this.random);
        this.positionalEmbedding = new PositionalEmbedding(blockSize, embeddingDimension, this.random);

        // Stack transformer blocks
        this.blocks = new ArrayList<>();
        for (int i = 0; i < numberOfLayers; i++) {
            this.blocks.add(new TransformerBlock(embeddingDimension, numHeads, useMultiHead, this.random));
        }

        this.embeddingNormalization = new RMSNormalization(embeddingDimension);
        // Final normalization + output
        this.finalNormalization = new RMSNormalization(embeddingDimension);
        this.outputHead = new Linear(embeddingDimension, vocabularySize, this.random);

        // Collect all parameters
        this.allParameters = new ArrayList<>();
        this.allParameters.addAll(this.tokenEmbedding.parameters());
        this.allParameters.addAll(this.positionalEmbedding.parameters());
        this.allParameters.addAll(this.embeddingNormalization.parameters());
        for (TransformerBlock block : this.blocks) {
            this.allParameters.addAll(block.parameters());
        }
        this.allParameters.addAll(this.finalNormalization.parameters());
        this.allParameters.addAll(this.outputHead.parameters());

        System.out.printf("GPT Model: vocabularySize=%d, blockSize=%d, embeddingDimension=%d, attention=%s, layers=%d%n",
                vocabularySize, blockSize, embeddingDimension,
                useMultiHead ? "multi-head(" + numHeads + ")" : "single-head",
                numberOfLayers);
        System.out.printf("Total parameters: %d%n", this.allParameters.size());
    }

    /**
     * Forward pass: context -> logits for next token.
     *
     * @param context Token IDs [blockSize]
     * @return Logits [vocabularySize] (from last position only)
     */
    public Value[] forward(int[] context) {
        final int sequenceLength = context.length;

        // Step 1: Token + positional embeddings combined
        Value[][] x = new Value[sequenceLength][this.embeddingDimension];
        for (int position = 0; position < sequenceLength; position++) {
            final Value[] tokenEmbedding = this.tokenEmbedding.forward(context[position]);
            final Value[] positionEmbedding = this.positionalEmbedding.forward(position);

            // Combine: element-wise addition
            for (int dimension = 0; dimension < this.embeddingDimension; dimension++) {
                x[position][dimension] = tokenEmbedding[dimension].add(positionEmbedding[dimension]);
            }
        }

        // Step 1b: Normalize embeddings
        x = this.embeddingNormalization.forward(x);

        // Step 2: Pass through all transformer blocks
        for (TransformerBlock block : this.blocks) {
            x = block.forward(x);
        }

        // Step 3: Final normalization
        x = this.finalNormalization.forward(x);

        // Step 4: Output logits from LAST position only.
        // The last position has attended to all previous positions.
        return this.outputHead.forward(x[sequenceLength - 1]);
    }

    /**
     * Compute cross-entropy loss for a training example.
     */
    public Value computeLoss(TrainingExample example) {
        final Value[] logits = forward(example.context());
        final Value[] probabilities = softmax(logits);
        return probabilities[example.target()].log().neg(); // -log(p_true)
    }

    /**
     * Forward + backward pass. Returns the loss.
     * The caller is responsible for zeroing gradients before and stepping the optimizer after.
     */
    public double trainStep(TrainingExample example) {
        final Value loss = computeLoss(example);
        double lossValue = loss.getData();
        loss.backward();
        return lossValue;
    }

    public List<Value> parameters() {
        return this.allParameters;
    }

    public String generate(CharacterTokenizer tokenizer, int maxLength, double temperature) {
        final List<Integer> generated = new ArrayList<>();
        final int bosId = tokenizer.getBOSId();

        // Start with BOS-filled context
        final int[] context = new int[this.blockSize];
        Arrays.fill(context, bosId);

        for (int step = 0; step < maxLength; step++) {
            // Get logits from last position
            final Value[] logits = forward(context);
            double[] logitData = new double[this.vocabularySize];
            for (int i = 0; i < vocabularySize; i++) {
                logitData[i] = logits[i].getData() / temperature;
            }

            // Sample next token
            double[] probabilities = softmaxDouble(logitData);
            int nextToken = sample(probabilities);

            if (nextToken == bosId) {
                break;
            }
            generated.add(nextToken);
            // Shift context window
            System.arraycopy(context, 1, context, 0, context.length - 1);
            context[context.length - 1] = nextToken;
        }
        return tokenizer.decode(generated);
    }

    private void zeroGrad() {
        for (Value parameter : this.allParameters) {
            parameter.setGradient(0.0);
        }
    }

    private Value[] softmax(Value[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (Value logit : logits) {
            max = Math.max(max, logit.getData());
        }

        Value[] exponents = new Value[logits.length];
        Value sum = new Value(0.0);
        for (int i = 0; i < logits.length; i++) {
            exponents[i] = logits[i].subtract(max).exp();
            sum = sum.add(exponents[i]);
        }

        Value[] probabilities = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = exponents[i].divide(sum);
        }
        return probabilities;
    }

    private double[] softmaxDouble(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }

        double sum = 0.0;
        double[] exponents = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exponents[i] = Math.exp(logits[i] - max);
            sum += exponents[i];
        }

        double[] probabilities = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = exponents[i] / sum;
        }
        return probabilities;
    }

    private int sample(double[] probabilities) {
        double r = this.random.nextDouble();
        double cdf = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cdf += probabilities[i];
            if (r <= cdf) {
                return i;
            }
        }
        return probabilities.length - 1;
    }
}
