package org.nadeemlab.impartial;

import org.json.JSONObject;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class DatasetPanel extends JPanel {
    private final ImpartialController controller;
    private DefaultListModel<String> listModel;
    private JList<String> list;
    private JRadioButton inferRadioButton;
    private JRadioButton entropyRadioButton;

    DatasetPanel(ImpartialController controller) {
        this.controller = controller;

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.PAGE_AXIS));

        JPanel viewSelector = createViewSelector();
        viewSelector.setAlignmentX(Component.LEFT_ALIGNMENT);

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("dataset"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        mainPanel.add(createSampleList());
        mainPanel.add(createViewSelector());

        add(mainPanel);
    }

    private JScrollPane createSampleList() {
        listModel = new DefaultListModel<>();

        //Create the list and put it in a scroll pane.
        list = new JList<>(listModel);
        list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        list.setSelectedIndex(0);
        list.setVisibleRowCount(5);

        JScrollPane listScroller = new JScrollPane(list);
        listScroller.setPreferredSize(new Dimension(200, 150));
        listScroller.setBorder(BorderFactory.createLineBorder(Color.lightGray));

        list.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                String imageId = getSelectedImageId();
                controller.updateImage(imageId);
            }
        });

        return listScroller;
    }

    private JPanel createViewSelector() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.LINE_AXIS));

        inferRadioButton = new JRadioButton("infer");
        inferRadioButton.addActionListener(e -> {
            controller.displayInfer();
//            thresholdSlider.setEnabled(true);
        });
        inferRadioButton.setEnabled(false);

        entropyRadioButton = new JRadioButton("entropy");
        entropyRadioButton.addActionListener(e -> {
            controller.displayEntropy();
//            thresholdSlider.setEnabled(false);
        });
        entropyRadioButton.setEnabled(false);

        ButtonGroup group = new ButtonGroup();
        group.add(inferRadioButton);
        group.add(entropyRadioButton);

        panel.add(inferRadioButton);
        panel.add(entropyRadioButton);

        return panel;
    }

    public void populateSampleList(String[] samples) {
        if (!listModel.isEmpty()) {
            listModel.clear();
        }

        Arrays.sort(samples);
        for (String sample : samples) {
            listModel.addElement(sample);
        }
    }

    public String getSelectedImageId() {
        return list.getSelectedValue();
    }

    public void updateInferView(boolean enable) {
        inferRadioButton.setEnabled(enable);
        entropyRadioButton.setEnabled(enable);
    }

}
