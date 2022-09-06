package org.nadeemlab.impartial;


import org.json.JSONObject;

import javax.swing.*;

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

        LabelPanel labelPanel = new LabelPanel(controller);
        mainPanel.add(labelPanel);

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

    public void updateInferInfo(int epoch, String time) {
        inferPanel.updateInferView(epoch, time);
    }

    public void disableInferInfo() {
        inferPanel.disableInferView();
    }

    public void updateInferView(boolean enable) {
        datasetPanel.updateInferView(enable);
    }

    public void enableInferButton() {
        inferPanel.enableInfer();
    }

    public float getThreshold() {
        return inferPanel.getThreshold();
    }

    public JSONObject getTrainParams() {
        return trainPanel.getTrainParams();
    }
}
