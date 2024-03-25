package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.util.Hashtable;

public class InferPanel extends JPanel {
    private final ImpartialController controller;
    private final JLabel thresholdValue;
    private JButton inferButton;
    private JButton batchInferButton;
    private JButton downloadButton;
    private JButton uploadButton;
    private final JSlider thresholdSlider = new JSlider(JSlider.HORIZONTAL, 0, 100, 50);

    InferPanel(ImpartialController controller) {
        this.controller = controller;

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Infer"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );
        setBorder(BorderFactory.createEmptyBorder(0, 5, 5, 5));

        JLabel title = new JLabel("Infer");
        title.setFont(new Font("sans-serif", Font.PLAIN, 15));
        add(title);
        add(Box.createVerticalStrut(10));

        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setAlignmentX(LEFT_ALIGNMENT);

        thresholdValue = new JLabel();
        thresholdValue.setAlignmentX(LEFT_ALIGNMENT);
        thresholdValue.setText(
                String.format("Threshold %.2f",  normalizeValue(thresholdSlider.getValue()))
        );

        panel.add(thresholdValue);
        panel.add(createThresholdSlider());

        add(createModelPanel());
        add(panel);
        add(createInferButtonPanel());
        add(createBatchInferButtonPanel());
    }

    private JPanel createThresholdSlider() {
        thresholdSlider.setAlignmentX(LEFT_ALIGNMENT);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<>();
        labelTable.put(0, new JLabel("0"));
        labelTable.put(100, new JLabel("1"));
        thresholdSlider.setLabelTable(labelTable);

        thresholdSlider.setPaintLabels(true);
        thresholdSlider.setMinorTickSpacing(10);
        thresholdSlider.setMajorTickSpacing(50);
        thresholdSlider.setPaintTicks(true);
        thresholdSlider.setPreferredSize(new Dimension(150, thresholdSlider.getPreferredSize().height));
        thresholdSlider.setEnabled(false);
        thresholdSlider.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));

        thresholdSlider.addChangeListener(e -> {
            JSlider source = (JSlider) e.getSource();
            int value = source.getValue();
            thresholdValue.setText(
                    String.format("Threshold %.2f",  normalizeValue(value))
            );
            controller.updateDisplay();
        });

        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
        sliderPanel.setAlignmentX(LEFT_ALIGNMENT);
        sliderPanel.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));
        sliderPanel.add(thresholdSlider);

        return sliderPanel;
    }

    private JPanel createModelPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
        panel.setAlignmentX(Component.LEFT_ALIGNMENT);
        panel.setBorder(BorderFactory.createEmptyBorder(0, 0, 5, 0));

        JLabel modelLabel = new JLabel("Model");

        downloadButton = new JButton("Download");
        downloadButton.addActionListener(e -> controller.downloadModelCheckpoint());
        downloadButton.setEnabled(false);

        uploadButton = new JButton("Upload");
        uploadButton.addActionListener(e -> controller.uploadModelCheckpoint());
        uploadButton.setEnabled(false);

        panel.add(modelLabel);
        panel.add(Box.createHorizontalGlue());
        panel.add(uploadButton);
        panel.add(downloadButton);

        return panel;
    }

    private JPanel createInferButtonPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 0, 0));
        panel.setAlignmentX(Component.LEFT_ALIGNMENT);

        inferButton = new JButton("Infer");
        inferButton.addActionListener(e -> controller.infer());
        inferButton.setEnabled(false);

        panel.add(inferButton);

        return panel;
    }

    private JPanel createBatchInferButtonPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 0, 0));
        panel.setAlignmentX(Component.LEFT_ALIGNMENT);

        batchInferButton = new JButton("Batch Infer");
        batchInferButton.addActionListener(e -> controller.batch_infer());
        batchInferButton.setEnabled(false);

        panel.add(batchInferButton);

        return panel;
    }


    private float normalizeValue(int value) {
        return (float) value / 100;
    }

    public float getThreshold() {
        return normalizeValue(thresholdSlider.getValue());
    }

    public void setEnabledInfer(boolean b) {
        inferButton.setEnabled(b);
        batchInferButton.setEnabled(b);
    }

    public void onStarted() {
        thresholdSlider.setEnabled(true);
        inferButton.setEnabled(true);
        batchInferButton.setEnabled(true);
        uploadButton.setEnabled(true);
        downloadButton.setEnabled(true);
    }

    public void onStopped() {
        thresholdSlider.setEnabled(false);
        inferButton.setEnabled(false);
        batchInferButton.setEnabled(false);
        uploadButton.setEnabled(false);
        downloadButton.setEnabled(false);
    }
}
