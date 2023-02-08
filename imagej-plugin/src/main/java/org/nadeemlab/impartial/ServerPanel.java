package org.nadeemlab.impartial;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.net.MalformedURLException;
import java.net.URL;

public class ServerPanel extends JPanel {
    private final JLabel statusLabel;
    private final JTextField monaiUrlTextField;
    private final JCheckBox requestServerCheckBox;
    private final JButton startButton;
    private final JButton stopButton;
    private String url = "http://localhost:8000";
    private boolean warningDisplayed = false;

    ServerPanel(ImpartialController controller) {

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("server"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JPanel content = new JPanel();
        content.setLayout(new BoxLayout(content, BoxLayout.PAGE_AXIS));

        monaiUrlTextField = new JTextField(url, 16);
        monaiUrlTextField.setAlignmentX(LEFT_ALIGNMENT);

        startButton = new JButton("start");
        startButton.setActionCommand("start");

        stopButton = new JButton("stop");
        stopButton.setEnabled(false);
        stopButton.setActionCommand("stop");

        requestServerCheckBox = new JCheckBox("request server");
        requestServerCheckBox.addActionListener(e -> {
            if (!warningDisplayed) {
                warningDisplayed = true;
                JOptionPane.showMessageDialog(controller.getContentPane(),
                        "This feature is currently available for MSK users only,\n" +
                                "but we're working on expanding it to everyone.",
                        "Restricted access",
                        JOptionPane.WARNING_MESSAGE
                );
            }
            startButton.setEnabled(true);

            if (requestServerCheckBox.isSelected()) {
                monaiUrlTextField.setEnabled(false);
                monaiUrlTextField.setText("https://impartial.mskcc.org");
            } else {
                monaiUrlTextField.setEnabled(true);
                monaiUrlTextField.setText(url);
            }
        });

        startButton.addActionListener(e -> {
            startButton.setEnabled(false);
            if (!requestServerCheckBox.isSelected())
                url = monaiUrlTextField.getText();

            controller.connect();
        });

        stopButton.addActionListener(e -> {
            startButton.setEnabled(false);
            controller.disconnect();
        });

        monaiUrlTextField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                startButton.setEnabled(true);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                startButton.setEnabled(true);
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startButton.setEnabled(true);
            }
        });

        JPanel innerPanel = new JPanel();
        innerPanel.setLayout(new BoxLayout(innerPanel, BoxLayout.LINE_AXIS));
        innerPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel lowInnerPanel = new JPanel();
        lowInnerPanel.setLayout(new BoxLayout(lowInnerPanel, BoxLayout.LINE_AXIS));
        lowInnerPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        statusLabel = new JLabel();
        statusLabel.setText("status: disconnected");

        innerPanel.add(startButton);
        innerPanel.add(requestServerCheckBox);
        innerPanel.add(Box.createHorizontalGlue());

        lowInnerPanel.add(stopButton);
        lowInnerPanel.add(statusLabel);
        lowInnerPanel.add(Box.createHorizontalGlue());

        content.add(monaiUrlTextField);
        content.add(innerPanel);
        content.add(lowInnerPanel);

        add(content);
    }

    public URL getUrl() throws MalformedURLException {
        return new URL(url);
    }

    public void setStatusLabel(String status) {
        statusLabel.setText(status);
    }

    public boolean getRequestServerCheckBox() {
        return requestServerCheckBox.isSelected();
    }

    public void onConnected() {
        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        requestServerCheckBox.setEnabled(false);
        statusLabel.setText("status: connected");
    }

    public void onDisconnected() {
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        requestServerCheckBox.setEnabled(true);
        statusLabel.setText("status: disconnected");
    }
}
