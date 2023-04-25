package org.nadeemlab.impartial;

import org.json.JSONObject;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableColumnModel;
import java.awt.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

public class ServerPanel extends JPanel {
    private final ImpartialController controller;
    private final JTextField monaiUrlTextField;
    private final JPanel sessionPanel;
    private JLabel sessionLabel;
    private JCheckBox requestServerCheckBox;
    private JButton startStopButton;
    private String url = "http://localhost:8000";
    private boolean warningDisplayed = false;
    private JButton selectSessionButton;

    ServerPanel(ImpartialController controller) {
        this.controller = controller;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("SERVER"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);

        monaiUrlTextField = new JTextField(url, 16);
        monaiUrlTextField.setAlignmentX(LEFT_ALIGNMENT);
        monaiUrlTextField.setText(url);

        sessionPanel = createSessionPanel();
        sessionPanel.setVisible(false);

        add(monaiUrlTextField);
        add(createButtonsPanel());
        add(sessionPanel);

        monaiUrlTextField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                startStopButton.setEnabled(true);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                startStopButton.setEnabled(true);
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startStopButton.setEnabled(true);
            }
        });
    }

    private void displayWarning() {
        String terms = "PLEASE READ THIS DOCUMENT CAREFULLY BEFORE YOU ACCESS OR USE IMPARTIAL. BY ACCESSING ANY PORTION OF IMPARTIAL, YOU AGREE TO BE BOUND BY THE TERMS AND CONDITIONS SET FORTH BELOW. IF YOU DO NOT WISH TO BE BOUND BY THESE TERMS AND CONDITIONS, PLEASE DO NOT ACCESS IMPARTIAL.\n" +
                "\n" +
                "ImPartial is developed and maintained by Memorial Sloan Kettering Cancer Center (“MSK,” “we”, or “us”) to provide an experimental artificial intelligence driven tool to perform segmentation using as few as 2-3 training images with some user-provided scribbles. MSK may, from time to time, update the software and other content on https://impartial.mskcc.org/ (“Content”). MSK makes no warranties or representations, and hereby disclaims any warranties, express or implied, with respect to any of the Content, including as to the present accuracy, completeness, timeliness, adequacy, or usefulness of any of the Content. By using this service, you agree that MSK will not be liable for any losses or damages arising from your use of or reliance on the Content, or other websites or information to which this website may be linked. \n" +
                "\n" +
                "By providing images to be analyzed using ImPartial (Submitted Images), you authorize MSK to copy, modify, display, distribute, perform, use, publish, and otherwise exploit the Submitted Images for any and all purposes, all without compensation to you, for as long as we decide (collectively, the \"Use Rights\"). In addition, you authorize us to grant any third party some or all of the Use Rights. By way of example, and not limitation, the Use Rights include the right for us to publish Submitted Images in whole or in part, and whether cropped, adopted, altered, or otherwise manipulated, for as long as we choose. By providing Submitted Images, you represent and warrant that (i) you own all rights in and to the Submitted Images (including any related copyrights or other intellectual property rights) or have sufficient authority and right to provide the content and to grant the Use Rights; (ii) your submission of the images and grant to us of Use Rights do not violate or conflict with the rights of other persons, or breach your obligations to other persons; and (iii) the Submitted Images do not include or contain any personally identifiable information (PII) or protected health information (PHI).\n" +
                "\n" +
                "DO NOT submit personally identifiable information (PII) or protected health information (PHI) in connection with any Submitted Images or otherwise.  \n" +
                "\n" +
                "You may use ImPartial the underlying Content, and images output therefrom for personal or academic research purposes only.  You may not use it for any other purpose, including not but limited to use for diagnosis, treatment, or patient care.  You may publish in scientific or academic journals or literature the results of such research, subject to providing credit to MSK as the source of output of ImPartial and with reference to these Terms of Use.  You may not otherwise redistribute or share the Content with any third party, in part or in whole, for any purpose, without the express written permission of MSK.\n" +
                "\n" +
                "Without limiting the generality of the foregoing, you may not use any part of the ImPartial, the underlying Content or the output for any other purpose, including:\n" +
                "\n" +
                "(i) use or incorporation into a commercial product or towards the performance of a commercial service;\n" +
                "(ii) research use in a commercial setting;\n" +
                "(ii) use for patient care or the provision of medical services; or\n" +
                "(iv) generation of reports in a medical, laboratory, hospital or other patient care setting.\n" +
                "\n" +
                "\n" +
                "You may not copy, transfer, reproduce, modify or create derivative works of ImPartial or the underlying Content for any commercial purpose without the express permission of MSK. \n" +
                "\n" +
                "The model output of ImPartial and the underlying Content is not a substitute for professional medical help, judgment or advice. Use of ImPartial does not create a physician-patient relationship or in any way make a person a patient of MSK. A physician or other qualified health provider should always be consulted for any health problem or medical condition. \n" +
                "\n" +
                "Neither these Terms or Use nor the availability of ImPartial should be understood to create an obligation or expectation that MSK will continue to make ImPartial available. MSK may discontinue or restrict the availability of ImPartial at any time. MSK may also modify these Terms or Use at any time.\n" +
                "\n" +
                "MSK respects the intellectual property rights of others, just as it expects others to respect its intellectual property. If you believe that any content (including Submitted Images and Content) on the website or other activity taking place on the website constitutes infringement of a work protected by copyright, please notify us at:\n" +
                "\n" +
                "cmsdigitalteam@mskcc.org\n" +
                "\n" +
                "Your notice must comply with the Digital Millennium Copyright Act (17 U.S.C. §512) (the \"DMCA\"). Upon receipt of a compliant notice, we will respond and proceed in accordance with the DMCA.\n" +
                "\n" +
                "By using ImPartial, you consent to the jurisdiction and venue of the state and federal courts located in New York City, New York, USA, for any claims related to or arising from your use of ImPartial or your violation of these Terms of Use and agree that you will not bring any claims against MSK that relate to or arise from the foregoing except in those courts. \n" +
                "\n" +
                "If any provision of these Terms of Use is held to be invalid or unenforceable, then such provision shall be struck, and the remaining provisions shall be enforced. Headings are for reference purposes only and in no way define, limit, construe, or describe the scope or extent of such section. MSK’s failure to act with respect to a breach by you or others does not waive its right to act with respect to subsequent or similar breaches. This agreement and the terms and conditions contained herein set forth the entire understanding and agreement between MSK and you with respect to the subject matter hereof and supersede any prior or contemporaneous understanding, whether written or oral.\n" +
                "\n" +
                "For inquiries about the Content, please contact us at nadeems@mskcc.org.\n" +
                "\n" +
                "If you are interested in using ImPartial for purposes beyond those permitted by these Terms of Use, please contact us at nadeems@mskcc.org to inquire concerning the availability of a license.\n" +
                "";

        JTextArea textArea = new JTextArea(terms);
        JScrollPane scrollPane = new JScrollPane(textArea);
        textArea.setLineWrap(true);
        textArea.setWrapStyleWord(true);
        scrollPane.setPreferredSize(new Dimension(500, 500));
        JOptionPane.showMessageDialog(null, scrollPane, "ImPartial Terms of Use",
                JOptionPane.WARNING_MESSAGE);
    }

    private JPanel createButtonsPanel() {
        JPanel panel = new JPanel();
        panel.setAlignmentX(LEFT_ALIGNMENT);
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

        startStopButton = new JButton("start");

        panel.add(createRequestServerCheckbox());
        panel.add(Box.createHorizontalGlue());
        panel.add(startStopButton);

        panel.add(Box.createVerticalGlue());

        startStopButton.addActionListener(e -> {
            selectSessionButton.setEnabled(false);
            startStopButton.setEnabled(false);
            if (startStopButton.getText().equals("start")) {
                if (!requestServerCheckBox.isSelected())
                    url = monaiUrlTextField.getText();
                controller.start();
            } else {
                controller.stop();
            }
        });

        return panel;
    }

    private JCheckBox createRequestServerCheckbox() {
        requestServerCheckBox = new JCheckBox("request server");

        requestServerCheckBox.addActionListener(e -> {
            if (!warningDisplayed) {
                warningDisplayed = true;
                displayWarning();
            }

            if (requestServerCheckBox.isSelected()) {
                monaiUrlTextField.setEnabled(false);
                monaiUrlTextField.setText("https://impartial.mskcc.org");
            } else {
                monaiUrlTextField.setEnabled(true);
                monaiUrlTextField.setText(url);
            }
        });

        return requestServerCheckBox;
    }

    private JPanel createSessionPanel() {
        JPanel panel = new JPanel();
        panel.setAlignmentX(LEFT_ALIGNMENT);
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

        sessionLabel = new JLabel("<html> <strong>session</strong>");

        selectSessionButton = new JButton("select");
        selectSessionButton.addActionListener(e -> {
            JSONObject sessions = controller.getSessions();

            ArrayList<UserSession> userSessions = new ArrayList<>();

            if (sessions.length() > 0) {
                String[] keys = JSONObject.getNames(sessions);
                for (String session_id : keys) {
                    JSONObject session = (JSONObject) sessions.get(session_id);

                    userSessions.add(new UserSession(
                            session_id,
                            session.getString("created_at"),
                            session.getJSONArray("images").length(),
                            session.getJSONArray("labels").length()
                    ));
                }
            }

            UserSessionDialog dialog = new UserSessionDialog(controller.getFrame(), userSessions);
            dialog.setVisible(true);
            if (dialog.getSelectedSession() != null) {
                controller.setSession(dialog.getSelectedSession());
            }
        });

        panel.add(sessionLabel);
        panel.add(Box.createHorizontalGlue());
        panel.add(selectSessionButton);

        return panel;
    }

    public void setSessionPanelVisible(boolean b) {
        sessionPanel.setVisible(b);
    }

    public void setSession(String sessionId) {
//        String sessionId = selectedSession.getId();
        sessionLabel.setText(String.format(
                "<html> <strong>session</strong> %s </html>",
                sessionId.substring(0, Math.min(sessionId.length(), 8))
        ));
    }

    public URL getUrl() throws MalformedURLException {
        return new URL(url);
    }

    public boolean getRequestServerCheckBox() {
        return requestServerCheckBox.isSelected();
    }

    public void onStarted() {
        startStopButton.setText("stop");
        startStopButton.setEnabled(true);
        selectSessionButton.setEnabled(true);
        requestServerCheckBox.setEnabled(false);
        if (requestServerCheckBox.isSelected()) {
            sessionPanel.setVisible(true);
            controller.getFrame().pack();
        }
    }

    public void onStopped() {
        startStopButton.setText("start");
        startStopButton.setEnabled(true);
        selectSessionButton.setEnabled(true);
        requestServerCheckBox.setEnabled(true);
        sessionLabel.setText("<html> <strong>session</strong> </html>");
        sessionPanel.setVisible(false);
        controller.getFrame().pack();
    }
}

class UserSessionDialog extends JDialog {
    private final JTable sessionTable;
    private final DefaultTableModel sessionTableModel;
    private UserSession selectedSession;

    public UserSessionDialog(Frame parent, ArrayList<UserSession> sessions) {
        super(parent, "Sessions", true);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        setSize(400, 300);

        sessionTableModel = new DefaultTableModel(new Object[]{"id", "date", "images", "labels"}, 0) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };

        sessionTable = new JTable(sessionTableModel);
        sessionTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        TableColumnModel columnModel = sessionTable.getColumnModel();
        columnModel.getColumn(0).setPreferredWidth(50);

        sessions.sort(Comparator.comparing(UserSession::getDate).reversed());
        for (UserSession session : sessions) {
            addSession(session);
        }

        JScrollPane scrollPane = new JScrollPane(sessionTable);

        JPanel panel = new JPanel(new BorderLayout());
        panel.add(scrollPane, BorderLayout.CENTER);

        JButton selectButton = new JButton("select");
        selectButton.addActionListener(e -> {
            int selectedRow = sessionTable.getSelectedRow();
            if (selectedRow != -1) {
                String sessionId = (String) sessionTableModel.getValueAt(selectedRow, 0);
                selectedSession = findSessionById(sessions, sessionId);
                dispose();
            }
        });

        JButton cancelButton = new JButton("cancel");
        cancelButton.addActionListener(e -> dispose());

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttonPanel.add(selectButton);
        buttonPanel.add(cancelButton);

        panel.add(buttonPanel, BorderLayout.SOUTH);

        setContentPane(panel);
        setLocationRelativeTo(parent);
    }

    private void addSession(UserSession session) {
        sessionTableModel.addRow(
                new Object[]{session.getId(), session.getParsedDate(), session.getNumImages(), session.getNumLabels()}
        );
    }

    private UserSession findSessionById(ArrayList<UserSession> sessions, String id) {
        for (UserSession session : sessions) {
            if (session.getId() == id) {
                return session;
            }
        }
        return null;
    }

    public UserSession getSelectedSession() {
        return selectedSession;
    }
}