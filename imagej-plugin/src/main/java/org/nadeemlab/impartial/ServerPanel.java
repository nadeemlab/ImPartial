package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.net.MalformedURLException;
import java.net.URL;

public class ServerPanel extends JPanel {
    private final JLabel statusLabel;
    private final JTextField monaiUrlTextField;
    private final JCheckBox requestServerCheckBox;
    private JButton connectButton;
    private String defaultUrl = "http://localhost:8000";

    ServerPanel(ImpartialController controller) {

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("server"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JPanel content = new JPanel();
        content.setLayout(new BoxLayout(content, BoxLayout.PAGE_AXIS));

        monaiUrlTextField = new JTextField(defaultUrl, 16);
        monaiUrlTextField.setAlignmentX(LEFT_ALIGNMENT);

        requestServerCheckBox = new JCheckBox("request server");
        requestServerCheckBox.addActionListener(e -> {
            connectButton.setEnabled(true);

            if (requestServerCheckBox.isSelected()) {
                monaiUrlTextField.setEnabled(false);
                monaiUrlTextField.setText("https://impartial.nadeemlab.org");
            } else {

                monaiUrlTextField.setEnabled(true);
                monaiUrlTextField.setText("http://localhost:8000");
            }
        });

        JPanel innerPanel = new JPanel();
        innerPanel.setLayout(new BoxLayout(innerPanel, BoxLayout.LINE_AXIS));
        innerPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        connectButton = new JButton("connect");
        connectButton.setActionCommand("start");

        connectButton.addActionListener(e -> {
            connectButton.setEnabled(false);
            controller.connect();
        });

        statusLabel = new JLabel();
        statusLabel.setText("status: disconnected");

        innerPanel.add(connectButton);
        innerPanel.add(statusLabel);
        innerPanel.add(Box.createHorizontalGlue());

        content.add(monaiUrlTextField);
        content.add(requestServerCheckBox);
        content.add(innerPanel);

        add(content);
    }

    public URL getUrl() {
        try {
            return new URL(monaiUrlTextField.getText());
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }

    public void setStatusLabel(String status) {
        statusLabel.setText(status);
    }

    public boolean getRequestServerCheckBox() {
        return requestServerCheckBox.isSelected();
    }

    public void onConnected() {
        connectButton.setEnabled(false);
        statusLabel.setText("status: connected");
    }
}
