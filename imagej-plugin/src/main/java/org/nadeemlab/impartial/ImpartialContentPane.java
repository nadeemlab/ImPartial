package org.nadeemlab.impartial;


import org.json.JSONObject;

import javax.swing.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;

public class ImpartialContentPane extends JPanel {
    private final ServerPanel serverPanel;
    private final DatasetPanel datasetPanel;
    private final InferPanel inferPanel;
    private final TrainPanel trainPanel;

    /**
     * Create the dialog.
     */
    public ImpartialContentPane(final ImpartialController controller) {
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setAlignmentX(LEFT_ALIGNMENT);

        serverPanel = new ServerPanel(controller);
        mainPanel.add(serverPanel);

        datasetPanel = new DatasetPanel(controller);
        mainPanel.add(datasetPanel);

        inferPanel = new InferPanel(controller);
        mainPanel.add(inferPanel);

        trainPanel = new TrainPanel(controller);
        mainPanel.add(trainPanel);

        add(mainPanel);
    }

    public void onStarted() {
        serverPanel.onStarted();
        datasetPanel.onStarted();
        inferPanel.onStarted();
        trainPanel.onStarted();
    }

    public void onStopped() {
        serverPanel.onStopped();
        datasetPanel.onStopped();
        inferPanel.onStopped();
        trainPanel.onStopped();
    }

    // Server
    public boolean getRequestServerCheckBox() {
        return serverPanel.getRequestServerCheckBox();
    }

    public URL getUrl() throws MalformedURLException {
        return serverPanel.getUrl();
    }

    public void setSession(String session_id) {
        serverPanel.setSession(session_id);
    }

    // Dataset
    public String getSelectedImageId() {
        return datasetPanel.getSelectedImageId();
    }

    public List<String> getSelectedViews() {
        return datasetPanel.getSelected();
    }

    public ListModel getListModel() {
        return datasetPanel.getListModel();
    }

    public void populateSampleList(JSONObject images) {
        datasetPanel.populateSampleList(images);
    }

    public void setEnabledInferAndEntropy(boolean enable) {
        datasetPanel.setEnabledInferAndEntropy(enable);
    }

    public void setSelectedInfer(boolean b) {
        datasetPanel.setSelectedInfer(b);
    }

    public void setEnabledLabel(boolean b) {
        datasetPanel.setEnabledLabel(b);
    }

    public void setSelectedLabel(boolean b) {
        datasetPanel.setSelectedLabel(b);
    }

    public void setEnabledSubmit(boolean b) {
        datasetPanel.setEnabledSubmit(b);
    }

    public void setSampleStatus(Sample sample, String status) {
        datasetPanel.setSampleStatus(sample, status);
    }

    public void sortList() {
        datasetPanel.getListModel().sort();
    }

    public void setSampleEntropy(Sample sample, double entropy) {
        datasetPanel.setSampleEntropy(sample, entropy);
    }

    public void setSelectedFirstImage() {
        datasetPanel.setSelectedFirstImage();
    }

    // Infer
    public void setEnabledInfer(boolean b) {
        inferPanel.setEnabledInfer(b);
    }

    public float getThreshold() {
        return inferPanel.getThreshold();
    }

    public void inferPerformed(int epoch, String time) {
        setEnabledInferAndEntropy(true);
        setSelectedInfer(true);
    }


    // Train
    public JSONObject getTrainParams() {
        return trainPanel.getTrainParams();
    }

    public void onTrainingStopped() {
        trainPanel.onTrainingStopped();
    }
}
