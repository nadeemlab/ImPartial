package org.nadeemlab.impartial;

import org.json.JSONObject;

import java.awt.*;
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
    private final static String newline = "\n";

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

        JScrollPane listScroller = new JScrollPane(list);
        listScroller.setPreferredSize(new Dimension(150, 80));
        listScroller.setBorder(BorderFactory.createLineBorder(Color.black));

        JPanel buttonsPanel = new JPanel();
        buttonsPanel.setLayout(new BoxLayout(buttonsPanel, BoxLayout.PAGE_AXIS));

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

        buttonsPanel.add(openButton);
        buttonsPanel.add(loadLabelButton);
        buttonsPanel.add(submitLabelButton);

        add(listScroller);
        add(buttonsPanel);
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

            controller.setImageId(imageId);
            openButton.setEnabled(true);
            loadLabelButton.setEnabled(hasLabels);
            submitLabelButton.setEnabled(hasLabels);
        }
    }
}
