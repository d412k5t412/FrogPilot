#pragma once

#include "selfdrive/frogpilot/ui/qt/offroad/frogpilot_settings.h"

class UtilitiesPanel : public FrogPilotListWidget {
  Q_OBJECT
public:
  explicit UtilitiesPanel(FrogPilotSettingsWindow *parent);

private:
  FrogPilotSettingsWindow *parent;

  Params params;
  Params paramsMemory{"/dev/shm/params"};

  FrogPilotButtonsControl *forceStartedBtn;
};
