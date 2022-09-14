package org.nadeemlab.impartial;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Collections;

public class ListModel extends AbstractListModel<Sample> {
    private ArrayList<Sample> samples = new ArrayList<>();

    @Override
    public int getSize() {
        return samples.size();
    }

    @Override
    public Sample getElementAt(int index) {
        return samples.get(index);
    }

    public void sort() {
        Collections.sort(samples);
        fireContentsChanged(this, 0, samples.size());
    }

    public boolean isEmpty() {
        return samples.isEmpty();
    }

    public void clear() {
        samples.clear();
    }

    public void addElement(Sample sample) {
        samples.add(sample);
        fireContentsChanged(this, samples.size(), samples.size());
    }

    public Sample get(int index) {
        return samples.get(index);
    }

    public void setElementAt(Sample sample, int index) {
        samples.set(index, sample);
        fireContentsChanged(this, index, index);
    }

    public int size() {
        return samples.size();
    }

    public int indexOf(Sample sample) {
        return samples.indexOf(sample);
    }
}
