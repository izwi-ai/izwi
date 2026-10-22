import { CreationField, CreationOption } from "@/components/creation/CreationParts";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LANGUAGE_OPTIONS } from "@/features/transcription/playground/support";

/** Detect the language: the request carries no hint. */
export const AUTO_LANGUAGE = "Auto";

/** The request's language hint: none for Auto. */
export function languageHint(language: string): string | undefined {
  return language === AUTO_LANGUAGE ? undefined : language;
}

/** The spoken-language picker every new transcript shows. */
export function SpeechTextLanguageField({
  value,
  onValueChange,
  disabled = false,
}: {
  value: string;
  onValueChange: (language: string) => void;
  disabled?: boolean;
}) {
  return (
    <CreationField label="Language">
      <Select value={value} onValueChange={onValueChange} disabled={disabled}>
        <SelectTrigger
          aria-label="Language"
          className="h-9 w-full rounded-[10px] border-(--border-default) bg-(--surface-panel) text-[13.5px]"
        >
          <SelectValue placeholder="Language" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={AUTO_LANGUAGE}>Auto-detect</SelectItem>
          {LANGUAGE_OPTIONS.map((language) => (
            <SelectItem key={language} value={language}>
              {language}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </CreationField>
  );
}

/** Off by default in every mode: a summary is only made when asked for. */
export function GenerateSummaryOption({
  checked,
  onCheckedChange,
  disabled = false,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <CreationOption
      title="Generate summary"
      description="Create an AI summary after the transcript is ready."
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
    />
  );
}
