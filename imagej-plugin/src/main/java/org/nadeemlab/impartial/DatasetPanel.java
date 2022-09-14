package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.*;
import java.util.List;

public class DatasetPanel extends JPanel implements ItemListener {
    private final ImpartialController controller;
    private ListModel listModel;
    private JList<Sample> list;
    private JCheckBox inferCheckBox;
    private JCheckBox entropyCheckBox;
    private JCheckBox labelCheckBox;
    private final JButton submitLabelButton;

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

        submitLabelButton = new JButton("submit label");
        submitLabelButton.setEnabled(false);
        submitLabelButton.addActionListener(e -> controller.submitLabel());

        mainPanel.add(createSampleList());
        mainPanel.add(createViewSelector());
        mainPanel.add(submitLabelButton);

        add(mainPanel);
    }

    private JScrollPane createSampleList() {
        listModel = new ListModel();

        //Create the list and put it in a scroll pane.
        list = new JList<>(listModel);
        list.setCellRenderer(new DatasetRenderer());
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

        labelCheckBox = new JCheckBox("label");
        labelCheckBox.setEnabled(false);

        inferCheckBox = new JCheckBox("infer");
        inferCheckBox.setEnabled(false);

        entropyCheckBox = new JCheckBox("entropy");
        entropyCheckBox.setEnabled(false);

        labelCheckBox.addItemListener(this);
        inferCheckBox.addItemListener(this);
        entropyCheckBox.addItemListener(this);

        panel.add(labelCheckBox);
        panel.add(inferCheckBox);
        panel.add(entropyCheckBox);

        return panel;
    }

    public void populateSampleList(String[] samples) {
        if (!listModel.isEmpty()) {
            listModel.clear();
        }

        Arrays.sort(samples);
        for (String sample : samples) {
            listModel.addElement(new Sample(sample, ""));
        }
    }

    public String getSelectedImageId() {
        return list.getSelectedValue().getName();
    }

    public void setEnabledInferAndEntropy(boolean enable) {
        inferCheckBox.setEnabled(enable);
        entropyCheckBox.setEnabled(enable);
    }

    public void setSelectedInfer(boolean b) {
        inferCheckBox.setSelected(b);
    }

    public void setEnabledLabel(boolean b) {
        labelCheckBox.setEnabled(b);
    }

    public List<String> getSelected() {
        List<String> selected = new ArrayList<>();

        if (labelCheckBox.isSelected()) selected.add(labelCheckBox.getText());
        if (inferCheckBox.isSelected()) selected.add(inferCheckBox.getText());
        if (entropyCheckBox.isSelected()) selected.add(entropyCheckBox.getText());

        return selected;
    }

    @Override
    public void itemStateChanged(ItemEvent e) {
        controller.updateDisplay();
    }

    public void setSelectedAll(boolean b) {
        labelCheckBox.setSelected(b);
        inferCheckBox.setSelected(b);
        entropyCheckBox.setSelected(b);
    }

    public void setEnabledSubmit(boolean b) {
        submitLabelButton.setEnabled(b);
    }

    public void setSampleStatus(Sample sample, String status) {
        int index = listModel.indexOf(sample);
        listModel.get(index).setStatus(status);
        listModel.setElementAt(sample, index);
    }

    public ListModel getListModel() {
        return listModel;
    }

    public void setSampleEntropy(Sample sample, double entropy) {
        int index = listModel.indexOf(sample);
        listModel.get(index).setEntropy(entropy);
        listModel.setElementAt(sample, index);
    }
}
