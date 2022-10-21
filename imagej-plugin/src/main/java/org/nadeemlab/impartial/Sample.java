package org.nadeemlab.impartial;


public class Sample implements Comparable<Sample>{
    private String name;
    private String status;
    private double entropy = 0;

    public Sample(String name, String status) {
        this.name = name;
        this.status = status;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void setEntropy(double entropy) {
        this.entropy = entropy;
    }

    @Override
    public int compareTo(Sample o) {
        return this.entropy > o.entropy ? -1 : 1;
    }
}
