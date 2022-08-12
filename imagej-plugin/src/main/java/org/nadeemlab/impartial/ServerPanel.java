package org.nadeemlab.impartial;

import javax.swing.*;
import javax.swing.plaf.BorderUIResource;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.MalformedURLException;
import java.net.URL;

public class ServerPanel extends JPanel {
    private ImpartialDialog controller;
    private MonaiLabelClient monaiClient;
    private JLabel statusLabel;
    private JTextField monaiUrl;

    private String defaultUrl = "http://10.0.3.117:8000";

    ServerPanel(ImpartialDialog controller, MonaiLabelClient monaiClient) {
        this.controller = controller;
        this.monaiClient = monaiClient;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("server"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JPanel content = new JPanel();
        content.setLayout(new BoxLayout(content, BoxLayout.PAGE_AXIS));

        JPanel addressBar = new JPanel();

        monaiUrl = new JTextField(defaultUrl, 15);

        JButton connectButton = new JButton("connect");
        connectButton.setActionCommand("connect");
        connectButton.addActionListener(new ConnectListener());

        addressBar.add(monaiUrl);
        addressBar.add(connectButton);

        statusLabel = new JLabel();
        statusLabel.setText("status: disconnected");
//        statusLabel.setIcon(new ImageIcon("check-circle.png"));

        content.add(addressBar);
        content.add(statusLabel);

        add(content);
    }

    class ConnectListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            URL url;
            try {
                url = new URL(monaiUrl.getText());
            } catch (MalformedURLException ex) {
                throw new RuntimeException(ex);
            }
            monaiClient.setUrl(url);
            try {
                monaiClient.getInfo();
                controller.connect();
                statusLabel.setText("status: connected");
            } catch (IllegalArgumentException ignored) {
                statusLabel.setText("status: disconnected");
            }
        }
    }

}
