package org.nadeemlab.impartial;

import org.json.JSONObject;

import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Arrays;
import java.util.stream.StreamSupport;
import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

public class DatasetPanel extends JPanel implements ListSelectionListener {
    private ImpartialDialog controller;
    private MonaiLabelClient monaiClient;
    private JList list;
    private JButton openButton;
    private JButton loadLabelButton;
    private JButton submitLabelButton;
    private DefaultListModel listModel;
    private JTextArea sampleInfo;
    private static final String hireString = "Hire";
    private static final String fireString = "Fire";
    private final static String newline = "\n";
    private JButton fireButton;
    private JTextField employeeName;

    DatasetPanel(ImpartialDialog controller, MonaiLabelClient monaiClient) {
        this.controller = controller;
        this.monaiClient = monaiClient;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("dataset"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        listModel = new DefaultListModel<String>();

        //Create the list and put it in a scroll pane.
        list = new JList(listModel);
        list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        list.setSelectedIndex(0);
        list.addListSelectionListener(this);
        list.setVisibleRowCount(5);

        JScrollPane listScrollPane = new JScrollPane(list);

        openButton = new JButton("open");
        openButton.setActionCommand("open");
        openButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.showImage();
            }
        });
        openButton.setEnabled(false);

        loadLabelButton = new JButton("load label");
        loadLabelButton.setActionCommand("load");
        loadLabelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.loadLabel();
            }
        });
        loadLabelButton.setEnabled(false);

        submitLabelButton = new JButton("submit label");
        submitLabelButton.setActionCommand("submit");
        submitLabelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.submitLabel();
            }
        });
        submitLabelButton.setEnabled(false);

        sampleInfo = new JTextArea(5, 20);
        sampleInfo.setEditable(false);
        sampleInfo.setOpaque(false);
        sampleInfo.setBorder(BorderFactory.createEmptyBorder());

        add(listScrollPane);
        add(sampleInfo);
        add(openButton);
        add(loadLabelButton);
        add(submitLabelButton);
    }

    public void populateSampleList() {
        if (!listModel.isEmpty()) {
            listModel.clear();
        }

        String[] samples = getDatastoreSamples();
        Arrays.sort(samples);
        for (String sample : samples) {
            listModel.addElement(sample);
        }
    }

    private void updateImageInfo(String imageName, boolean hasLabels) {
        this.sampleInfo.setText("name: " + imageName + newline +
                "labeled: " + (hasLabels ? "yes" : "no"));
    }

    private JSONObject getSampleInfo(String sampleId) {
        JSONObject datastore = monaiClient.getDatastore();

        return datastore.getJSONObject("objects").getJSONObject(sampleId);
    }

    private String[] getDatastoreSamples() {
        JSONObject datastore = monaiClient.getDatastore();
        Iterable<String> iterable = () -> datastore.getJSONObject("objects").keys();

        return StreamSupport.stream(iterable.spliterator(), false)
                .toArray(String[]::new);
    }

    //This method is required by ListSelectionListener.
    public void valueChanged(ListSelectionEvent e) {
        if (e.getValueIsAdjusting() == false) {
            String imageId = (String) list.getSelectedValue();
            JSONObject sampleInfo = getSampleInfo(imageId);

            String imageName = sampleInfo.getJSONObject("image")
                    .getJSONObject("info")
                    .getString("name");
            boolean hasLabels = sampleInfo.has("labels");

            updateImageInfo(imageName, hasLabels);

            controller.setImageId(imageId);
            openButton.setEnabled(true);
            loadLabelButton.setEnabled(hasLabels);
            submitLabelButton.setEnabled(hasLabels);
        }
    }
}
