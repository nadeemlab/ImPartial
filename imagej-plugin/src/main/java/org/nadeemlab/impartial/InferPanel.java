package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.util.Hashtable;

public class InferPanel extends JPanel {
    private ImpartialController controller;
    private JButton inferButton;
    private JLabel trainInfo;
    private JSlider thresholdSlider = new JSlider(JSlider.HORIZONTAL, 0, 100, 50);

    InferPanel(ImpartialController controller) {
        this.controller = controller;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("infer"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        add(createInferPanel());
    }

    private JPanel createInferPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.PAGE_AXIS));

        JPanel thresholdSlider = createThresholdSlider();

        inferButton = new JButton("infer");
        inferButton.addActionListener(e -> controller.infer());
        inferButton.setEnabled(false);
        inferButton.setAlignmentX(Component.LEFT_ALIGNMENT);

        trainInfo = new JLabel("last run never");
        trainInfo.setEnabled(false);
        trainInfo.setAlignmentX(Component.LEFT_ALIGNMENT);

        panel.add(thresholdSlider);
        panel.add(trainInfo);
        panel.add(inferButton);

        return panel;
    }

    private JPanel createThresholdSlider() {
        JPanel panel = new JPanel();
        panel.setAlignmentX(LEFT_ALIGNMENT);
        panel.setLayout(new BoxLayout(panel, BoxLayout.PAGE_AXIS));

        JLabel thresholdValue = new JLabel();
        thresholdValue.setAlignmentX(LEFT_ALIGNMENT);
        thresholdValue.setText("threshold " + normalizeValue(thresholdSlider.getValue()));

        JPanel sliderPanel = new JPanel();
        sliderPanel.setAlignmentX(LEFT_ALIGNMENT);

        Hashtable<Integer, JLabel> labelTable = new Hashtable<>();
        labelTable.put(0, new JLabel("0"));
        labelTable.put(100, new JLabel("1"));
        thresholdSlider.setLabelTable(labelTable);

        thresholdSlider.setPaintLabels(true);
        thresholdSlider.setMinorTickSpacing(10);
        thresholdSlider.setMajorTickSpacing(50);
        thresholdSlider.setPaintTicks(true);

        thresholdSlider.addChangeListener(e -> {
            JSlider source = (JSlider) e.getSource();
            int value = source.getValue();
            thresholdValue.setText("threshold " + normalizeValue(value));
            controller.displayInfer();
        });

        thresholdSlider.setPreferredSize(new Dimension(150, 50));

        sliderPanel.add(thresholdSlider);

        panel.add(thresholdValue);
        panel.add(sliderPanel);

        return panel;
    }

    private float normalizeValue(int value) {
        return (float) value / 100;
    }

    public float getThreshold() {
        return normalizeValue(thresholdSlider.getValue());
    }

    public void updateInferView(int epoch, String time) {
//        inferRadioButton.setEnabled(true);
//        entropyRadioButton.setEnabled(true);

        trainInfo.setText("last run " + time + ", epoch " + epoch);
    }

    public void disableInferView() {
//        inferRadioButton.setEnabled(false);
//        entropyRadioButton.setEnabled(false);

        trainInfo.setText("last run never");
    }

    public void enableInfer() {
        inferButton.setEnabled(true);
    }
}
