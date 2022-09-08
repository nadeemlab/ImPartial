package org.nadeemlab.impartial;


import org.json.JSONObject;

import javax.swing.*;
import java.util.List;

public class ImpartialContentPane extends JPanel {
    ImpartialController controller;
    private final DatasetPanel datasetPanel;
    private final TrainPanel trainPanel;
    private final InferPanel inferPanel;

    /**
     * Create the dialog.
     */
    public ImpartialContentPane(final ImpartialController controller) {
        this.controller = controller;
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.PAGE_AXIS));

        ServerPanel serverPanel = new ServerPanel(controller);
        mainPanel.add(serverPanel);

        datasetPanel = new DatasetPanel(controller);
        mainPanel.add(datasetPanel);

        inferPanel = new InferPanel(controller);
        mainPanel.add(inferPanel);

        trainPanel = new TrainPanel(controller);
        mainPanel.add(trainPanel);

        add(mainPanel);
    }

    public void populateSampleList(String[] samples) {
        datasetPanel.populateSampleList(samples);
    }

    public String getSelectedImageId() {
        return datasetPanel.getSelectedImageId();
    }

    public void setEnabledInferAndEntropy(boolean enable) {
        datasetPanel.setEnabledInferAndEntropy(enable);
    }

    public void setSelectedInfer(boolean b) {
        datasetPanel.setSelectedInfer(b);
    }

    public void setEnabledInfer(boolean b) {
        inferPanel.setEnabledInfer(b);
    }

    public float getThreshold() {
        return inferPanel.getThreshold();
    }

    public JSONObject getTrainParams() {
        return trainPanel.getTrainParams();
    }

    public void inferPerformed(int epoch, String time) {
        setTextInfer("last run " + time + ", epoch " + epoch);
        setEnabledInferAndEntropy(true);
        setSelectedInfer(true);
    }

    public List<String> getSelectedViews() {
        return datasetPanel.getSelected();
    }

    public void setEnabledLabel(boolean b) {
        datasetPanel.setEnabledLabel(b);
    }

    public void setTextInfer(String s) {
        inferPanel.setTextInfer(s);
    }

    public void setSelectedAll(boolean b) {
        datasetPanel.setSelectedAll(b);
    }

    public void setEnabledSubmit(boolean b) {
        datasetPanel.setEnabledSubmit(b);
    }
}
