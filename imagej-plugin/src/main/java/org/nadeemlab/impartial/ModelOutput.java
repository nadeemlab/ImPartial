package org.nadeemlab.impartial;

import ij.process.FloatProcessor;

public class ModelOutput {
    private FloatProcessor output;
    private FloatProcessor entropy;
    private String time;
    private int epoch;

    public ModelOutput(FloatProcessor output, FloatProcessor entropy, String time, int epoch) {
        this.output = output;
        this.entropy = entropy;
        this.time = time;
        this.epoch = epoch;
    }

    public FloatProcessor getOutput() {
        return output;
    }

    public FloatProcessor getEntropy() {
        return entropy;
    }

    public String getTime() {
        return time;
    }

    public int getEpoch() {
        return epoch;
    }
}
