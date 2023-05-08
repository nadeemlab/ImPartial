package org.nadeemlab.impartial;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableColumnModel;
import java.awt.*;
import java.util.ArrayList;
import java.util.Comparator;

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