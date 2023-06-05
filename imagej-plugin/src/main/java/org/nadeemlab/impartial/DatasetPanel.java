package org.nadeemlab.impartial;


import org.json.JSONObject;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.StreamSupport;

public class DatasetPanel extends JPanel implements ItemListener {
    private final ImpartialController controller;
    private JButton submitLabelButton;
    private ListModel listModel;
    private JList<Sample> list;
    private JCheckBox inferCheckBox;
    private JCheckBox entropyCheckBox;
    private JCheckBox labelCheckBox;
    private JButton uploadButton;
    private JButton deleteButton;
    private final JLabel channelCountLabel;
    private final JLabel labelCountLabel;
    private final JLabel imageCountLabel;

    DatasetPanel(ImpartialController controller) {
        this.controller = controller;

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);
        setBorder(BorderFactory.createEmptyBorder(0, 5, 5, 5));

        JLabel title = new JLabel("Dataset");
        title.setFont(new Font("sans-serif", Font.PLAIN, 15));
        add(title);
        add(Box.createVerticalStrut(10));

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

        JPanel viewSelector = createViewSelector();
        viewSelector.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel innerPanel = new JPanel(new GridBagLayout());
        innerPanel.setAlignmentX(LEFT_ALIGNMENT);

        channelCountLabel = new JLabel("0");
        labelCountLabel = new JLabel("0");
        imageCountLabel = new JLabel("0");

        addRow(innerPanel, "Channels", channelCountLabel);
        addRow(innerPanel, "Labels", labelCountLabel, createSubmitButton());
        addRow(innerPanel, "Images", imageCountLabel, createUploadButton(), createDeleteButton());

        mainPanel.add(innerPanel);

        mainPanel.add(createSampleList());
        mainPanel.add(createViewSelector());

        add(mainPanel);
    }

    private JButton createSubmitButton() {
        submitLabelButton = new JButton("Submit");
        submitLabelButton.setEnabled(false);
        submitLabelButton.addActionListener(e -> controller.submitLabel());

        return submitLabelButton;
    }

    private JButton createDeleteButton() {
        deleteButton = new JButton("Delete");
        deleteButton.setEnabled(false);
        deleteButton.addActionListener(e -> controller.deleteSelectedImage());

        return deleteButton;
    }

    private JButton createUploadButton() {
        uploadButton = new JButton("Add");
        uploadButton.setEnabled(false);
        uploadButton.addActionListener(e -> controller.uploadImages());

        return uploadButton;
    }

    private void addRow(JPanel panel, String labelText, JLabel valueLabel, JButton... buttons) {
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = GridBagConstraints.RELATIVE;
        c.anchor = GridBagConstraints.LINE_START;
        c.insets = new Insets(0, 0, 5, 0);

        JLabel label = new JLabel(labelText);
        panel.add(label, c);

        c.gridx = 1;
        c.weightx = 1;
        c.anchor = GridBagConstraints.LINE_START;
        c.insets = new Insets(0, 5, 5, 0);

        panel.add(valueLabel, c);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 0, 0));
        for (JButton button : buttons) {
            buttonPanel.add(button);
        }
        c.gridx = 2;
        c.weightx = 1;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.insets = new Insets(0, 0, 5, 0);
        panel.add(buttonPanel, c);
    }

    private JPanel createSampleList() {
        JPanel panel = new JPanel();
        panel.setAlignmentX(LEFT_ALIGNMENT);
        panel.setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));

        listModel = new ListModel();

        //Create the list and put it in a scroll pane.
        list = new JList<>(listModel);
        list.setCellRenderer(new DatasetRenderer());
        list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        list.setSelectedIndex(0);
        list.setVisibleRowCount(5);

        JScrollPane listScroller = new JScrollPane(list);
        listScroller.setAlignmentX(LEFT_ALIGNMENT);
        listScroller.setPreferredSize(new Dimension(240, 150));
        listScroller.setBorder(BorderFactory.createLineBorder(Color.lightGray));

        list.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                controller.updateImage();
            }
        });

        panel.add(listScroller);

        return panel;
    }

    private JPanel createViewSelector() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        panel.setAlignmentX(Component.LEFT_ALIGNMENT);

        labelCheckBox = new JCheckBox("Label");
        labelCheckBox.setEnabled(false);

        inferCheckBox = new JCheckBox("Infer");
        inferCheckBox.setEnabled(false);

        entropyCheckBox = new JCheckBox("Entropy");
        entropyCheckBox.setEnabled(false);

        labelCheckBox.addItemListener(this);
        inferCheckBox.addItemListener(this);
        entropyCheckBox.addItemListener(this);

        panel.add(labelCheckBox);
        panel.add(inferCheckBox);
        panel.add(entropyCheckBox);

        return panel;
    }

    public void populateSampleList(JSONObject images) {
        if (!listModel.isEmpty()) {
            listModel.clear();
        }

        Iterable<String> iterable = images::keys;
        String[] samples = StreamSupport.stream(iterable.spliterator(), false)
                .toArray(String[]::new);

        Arrays.sort(samples);
        for (String sample : samples) {
            listModel.addElement(new Sample(sample, ""));
        }

        imageCountLabel.setText(String.valueOf(images.length()));

        int labelsCount = 0;
        String[] keys = JSONObject.getNames(images);
        if (keys != null) {
            for (String key : keys) {
                if (images.getJSONObject(key).getJSONObject("labels").length() > 0)
                    labelsCount++;
            }
        }
        labelCountLabel.setText(String.valueOf(labelsCount));

        String numberOfChannels = String.valueOf(controller.getNumberOfChannels());
        channelCountLabel.setText(numberOfChannels);
    }

    public String getSelectedImageId() {
        if (list.isSelectionEmpty())
            return null;
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
        if (listModel.isEmpty())
            return;
        controller.updateDisplay();
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

    public void setSelectedFirstImage() {
        if (!listModel.isEmpty())
            list.setSelectedIndex(0);
    }

    public void onStarted() {
        uploadButton.setEnabled(true);
        deleteButton.setEnabled(true);

        JSONObject datastore = controller.getDatastore();
        populateSampleList(datastore.getJSONObject("objects"));

        setSelectedFirstImage();
    }

    public void setSelectedLabel(boolean b) {
        labelCheckBox.setSelected(b);
    }

    public void onStopped() {
        listModel.clear();
        setSelectedAll(false);
        setEnabledAll(false);
        uploadButton.setEnabled(false);
        deleteButton.setEnabled(false);
        submitLabelButton.setEnabled(false);

        imageCountLabel.setText("0");
        labelCountLabel.setText("0");
        channelCountLabel.setText("0");
    }

    public void setSelectedAll(boolean b) {
        labelCheckBox.setSelected(b);
        inferCheckBox.setSelected(b);
        entropyCheckBox.setSelected(b);
    }

    public void setEnabledAll(boolean b) {
        labelCheckBox.setEnabled(b);
        inferCheckBox.setEnabled(b);
        entropyCheckBox.setEnabled(b);
    }
}


